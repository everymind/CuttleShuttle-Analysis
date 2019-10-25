import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import sys
import itertools
import scipy
import scipy.signal
from scipy import stats
import random

### FUNCTIONS ###
def categorize_by_animal(TGB_files):
    all_animals_dict = {}
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        all_animals_dict.setdefault(TGB_animal,[]).append(TGB_moment)
    return all_animals_dict

def categorize_by_animal_catchVmiss(TGB_files):
    catch_dict = {}
    miss_dict = {}
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        if TGB_type == "catch":
            catch_dict.setdefault(TGB_animal,[]).append(TGB_moment)
        if TGB_type == "miss": 
            miss_dict.setdefault(TGB_animal,[]).append(TGB_moment)
    return catch_dict, miss_dict

def filtered_basesub_count(TS_dict, prey_type, baseline_len, savgol_filter_window):
    basesub_filtered_TS = {}
    # make baseline for each animal, catch vs miss
    for animal in TS_dict: 
        basesub_filtered_TS[animal] = {}
        try:
            # baseline subtract each trial, then apply sav-gol filter
            all_filtered_basesub_trials = []
            for trial in TS_dict[animal]:
                filtered_trial = scipy.signal.savgol_filter(trial, savgol_filter_window, 3)
                baseline = np.nanmean(filtered_trial[0:baseline_len])
                filtered_basesub_trial = [float(x-baseline) for x in filtered_trial]
                all_filtered_basesub_trials.append(filtered_basesub_trial)
            basesub_filtered_mean_byTB = np.nanmean(all_filtered_basesub_trials, axis=0)
            basesub_filtered_mean_bySess = np.nanmean(all_filtered_basesub_trials)
            basesub_filtered_std_byTB = np.nanstd(all_filtered_basesub_trials, axis=0, ddof=1)
            basesub_filtered_std_bySess = np.nanstd(all_filtered_basesub_trials, ddof=1)
            basesub_filtered_TS[animal]['trials'] = all_filtered_basesub_trials
            basesub_filtered_TS[animal]['mean tb'] = basesub_filtered_mean_byTB
            basesub_filtered_TS[animal]['mean session'] = basesub_filtered_mean_bySess
            basesub_filtered_TS[animal]['std tb'] = basesub_filtered_std_byTB
            basesub_filtered_TS[animal]['std session'] = basesub_filtered_std_bySess
        except Exception:
            print("{a} made no tentacle shots during {p} prey movement type".format(a=animal, p=prey_type))
    return basesub_filtered_TS

def zScored_count(Zscore_type, dict_to_Zscore, dict_for_mean_std):
    zScored_dict = {}
    for animal in dict_to_Zscore:
        zScored_dict[animal] = []
        for trial in dict_to_Zscore[animal]['trials']:
            trial_array = np.array(trial)
            if Zscore_type=='timebin':
                trial_zscored = (trial_array - dict_for_mean_std[animal]['mean tb'])/dict_for_mean_std[animal]['std tb']
            if Zscore_type=='session':
                trial_zscored = []
                for timebin in trial:
                    tb_zscored = (timebin - dict_for_mean_std[animal]['mean session'])/dict_for_mean_std[animal]['std session']
                    trial_zscored.append(tb_zscored)
            zScored_dict[animal].append(trial_zscored)
    return zScored_dict

def shuffle_test(Group1, Group2, N_Shuffles, Group1_str, Group2_str, Group1_N, Group2_N, plot_on, plots_dir, todays_dt):
    # Observed performance
    OPerf = np.nanmean(Group1) - np.nanmean(Group2)
    # Shuffle the dataset and compare means again
    num_of_shuffles = N_Shuffles
    SPerf = np.zeros((num_of_shuffles,1))
    All_Group = np.concatenate([Group1, Group2])
    for shuff in range(num_of_shuffles):
        shuff_response = np.random.permutation(All_Group)
        SPerf[shuff] = np.nanmean(shuff_response[0:len(Group1)]) - np.nanmean(shuff_response[len(Group1):])
    # p-value of shuffle test
    pVal = np.nanmean(SPerf**2 >= OPerf**2)
    # sigma
    shuffled_mean = np.nanmean(SPerf)
    sigma_shuff = np.nanstd(SPerf, ddof=1)
    shuff_975p = np.percentile(SPerf, 97.5)
    shuff_025p = np.percentile(SPerf, 2.5)
    if plot_on == True:
        # show histogram of diffs of shuffled means
        figure_name = 'ShuffleTest_'+ Group1_str + '_' + Group2_str + '_' + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = "Histogram of the differences in means of randomly labeled data, Number of shuffles = {Ns}\n Group 1: {G1}, N = {G1N}\n Group 2: {G2}, N = {G2N}\n P-value of shuffle test: {p:.4f}, Mean of shuffle test: {m:.4f}, Sigma of shuffle test: {s:.4f}".format(Ns=N_Shuffles, G1=Group1_str, G1N=Group1_N, G2=Group2_str, G2N=Group2_N, p=pVal, m=shuffled_mean, s=sigma_shuff)
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        plt.hist(SPerf)
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.plot((shuff_025p, shuff_025p), (ymin, ymax/2), 'r-', linewidth=1)
        plt.plot(shuff_025p, ymax/2, 'ro')
        plt.text(shuff_025p, ymax/2-ymax/20, '2.5 percentile:\n'+'%.4f'%(shuff_025p), fontsize='x-small', ha='right', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
        plt.plot((shuff_975p, shuff_975p), (ymin, ymax/2), 'r-', linewidth=1)
        plt.plot(shuff_975p, ymax/2, 'ro')
        plt.text(shuff_975p, ymax/2-ymax/20, '97.5 percentile:\n'+'%.4f'%(shuff_975p), fontsize='x-small', ha='left', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
        plt.plot((OPerf, OPerf), (ymin, ymax), 'g--', linewidth=1)
        plt.text(OPerf, ymax-5, "Difference of Labeled Means = " + str(OPerf), fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    return SPerf, pVal, shuffled_mean

def pool_acrossA_acrossTB(catches_dict, misses_dict, timebin_start, timebin_end, prey_type_str):
    pooled_catches = []
    pooled_catches_Ntrials = 0
    pooled_misses = []
    pooled_misses_Ntrials = 0
    for animal in catches_dict:
        thisA_catchesN = len(catches_dict[animal])
        pooled_catches_Ntrials = pooled_catches_Ntrials + thisA_catchesN
        if thisA_catchesN != 0:
            for trial in catches_dict[animal]:
                if timebin_end == -1:
                    for binned_count in trial[timebin_start:]:
                        pooled_catches.append(binned_count)
                else:
                    for binned_count in trial[timebin_start:timebin_end]:
                        pooled_catches.append(binned_count)
        else:
            print('{a} made no catch tentacle shots during {p} prey movement'.format(a=animal, p=prey_type_str))
        thisA_missesN = len(misses_dict[animal])
        pooled_misses_Ntrials = pooled_misses_Ntrials + thisA_missesN
        if thisA_missesN != 0:
            for trial in misses_dict[animal]:
                if timebin_end == -1:
                    for binned_count in trial[timebin_start:]:
                        pooled_misses.append(binned_count)
                else:
                    for binned_count in trial[timebin_start:timebin_end]:
                        pooled_misses.append(binned_count)
        else:
            print('{a} made no miss tentacle shots during {p} prey movement'.format(a=animal, p=prey_type_str))
    pooled_catches_array = np.array(pooled_catches)
    pooled_misses_array = np.array(pooled_misses)
    return pooled_catches_array, pooled_catches_Ntrials, pooled_misses_array, pooled_misses_Ntrials

def pool_timebins_byAnimal(catches_dict, misses_dict, start_tb, end_tb):
    pooledTB_byA_catches = {}
    pooledTB_byA_misses = {}
    for animal in catches_dict:
        pooledTB_byA_catches[animal] = []
        pooledTB_byA_misses[animal] = []
        for trial in catches_dict[animal]:
            if end_tb == -1:
                for binned_count in trial[start_tb:]:
                    pooledTB_byA_catches[animal].append(binned_count)
            else:    
                for binned_count in trial[start_tb:end_tb]:
                    pooledTB_byA_catches[animal].append(binned_count)
        for trial in misses_dict[animal]:
            if end_tb == -1: 
                for binned_count in trial[start_tb:]:
                    pooledTB_byA_misses[animal].append(binned_count)
            else:    
                for binned_count in trial[start_tb:end_tb]:
                    pooledTB_byA_misses[animal].append(binned_count)
    return pooledTB_byA_catches, pooledTB_byA_misses

def pool_acrossA_keepTemporalStructure(catches_dict, misses_dict, timebin_start, timebin_end, prey_type_str):
    pooled_catches = []
    pooled_catches_Ntrials = 0
    pooled_misses = []
    pooled_misses_Ntrials = 0
    for animal in catches_dict:
        thisA_catchesN = len(catches_dict[animal])
        pooled_catches_Ntrials = pooled_catches_Ntrials + thisA_catchesN
        if thisA_catchesN != 0:
            for trial in catches_dict[animal]:
                if timebin_end == -1:
                    pooled_catches.append(trial[timebin_start:])
                else:     
                    pooled_catches.append(trial[timebin_start:timebin_end])
        else:
            print('{a} made no catch tentacle shots during {p} prey movement'.format(a=animal, p=prey_type_str))
        thisA_missesN = len(misses_dict[animal])
        pooled_misses_Ntrials = pooled_misses_Ntrials + thisA_missesN
        if thisA_missesN != 0:
            for trial in misses_dict[animal]:
                if timebin_end == -1:
                    pooled_misses.append(trial[timebin_start:])
                else:     
                    pooled_misses.append(trial[timebin_start:timebin_end])
        else:
            print('{a} made no miss tentacle shots during {p} prey movement'.format(a=animal, p=prey_type_str))
    pooled_catches_array = np.array(pooled_catches)
    pooled_misses_array = np.array(pooled_misses)
    return pooled_catches_array, pooled_catches_Ntrials, pooled_misses_array, pooled_misses_Ntrials

def plot_indiv_animals(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_C_dict, allA_M_dict, TGB_bucket, baseline_len, plots_dir, todays_dt):
    # plot individual animals
    img_type = ['.png', '.pdf']
    for animal in allA_C_dict.keys(): 
        try:
            if 'Zscored' in preprocess_str:
                N_catch = len(allA_C_dict[animal])
                N_miss = len(allA_M_dict[animal])
                catches_mean = np.nanmean(allA_C_dict[animal], axis=0)
                misses_mean = np.nanmean(allA_M_dict[animal], axis=0)
                # set fig path and title
                if len(prey_type_str.split(' '))>1:
                    figure_name = analysis_type_str +'_'+ preprocess_str +'_'+ prey_type_str.split(' ')[1] + 'Trials_' + animal + "_" + todays_dt + img_type[0]
                else:
                    figure_name = analysis_type_str +'_'+ preprocess_str +'_'+ prey_type_str + 'Trials_' + animal + "_" + todays_dt + img_type[0]
                figure_path = os.path.join(plots_dir, figure_name)
                figure_title = 'Z-scored mean change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Individual trials plotted with more transparent traces \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, Animal: {a}\n Number of catches: {Nc}, Number of misses: {Nm}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, a=animal, Nc=str(N_catch), Nm=str(N_miss))
                # setup fig
                plt.figure(figsize=(16,9), dpi=200)
                plt.suptitle(figure_title, fontsize=12, y=0.99)
                plt.ylabel("Change from baseline in number of edges")
                plot_xticks = np.arange(0, len(allA_C_dict[animal][0]), step=60)
                plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
                #plt.xlim(0,180)
                plt.ylim(-6, 6)
                plt.xlabel("Seconds")
                plt.grid(b=True, which='major', linestyle='-')
                # plot z-scored edge counts
                for trial in allA_M_dict[animal]:
                    plt.plot(trial, linewidth=1, color=[1.0, 0.0, 0.0, 0.1])
                for trial in allA_C_dict[animal]:
                    plt.plot(trial, linewidth=1, color=[0.0, 0.0, 1.0, 0.1])
                plt.plot(misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
                #plt.fill_between(range(len(allA_M_dict_mean[animal])), misses_mean-canny_std_miss, misses_mean+canny_std_miss, color=[1.0, 0.0, 0.0, 0.1])
                plt.plot(catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
                #plt.fill_between(range(len(allA_C_dict_mean[animal])), catches_mean-canny_std_catch, catches_mean+canny_std_catch, color=[0.0, 0.0, 1.0, 0.1])
                # plot events
                ymin, ymax = plt.ylim()
                plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
                plt.text(baseline_len, ymax-0.8, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
                plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
                plt.text(TGB_bucket, ymax-0.5, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
                plt.legend(loc='upper left')
                # save fig
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
            else:
                if animal in allA_C_dict:
                    N_catch = len(allA_C_dict[animal]['trials'])
                    catches_mean = np.nanmean(allA_C_dict[animal]['trials'], axis=0)
                if animal in allA_M_dict:
                    N_miss = len(allA_M_dict[animal]['trials'])
                    misses_mean = np.nanmean(allA_M_dict[animal]['trials'], axis=0)
                # set fig path and title
                figure_name = analysis_type_str +'_'+ preprocess_str +'_'+ prey_type_str.split(' ')[1] + 'Trials_' + animal + "_" + todays_dt + img_type[0]
                figure_path = os.path.join(plots_dir, figure_name)
                figure_title = 'SavGol filtered and baseline subtracted mean change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Individual trials plotted with more transparent traces \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, Animal: {a}\n Number of catches: {Nc}, Number of misses: {Nm}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, a=animal, Nc=str(N_catch), Nm=str(N_miss))
                # setup fig
                plt.figure(figsize=(16,9), dpi=200)
                plt.suptitle(figure_title, fontsize=12, y=0.99)
                plt.ylabel("Change from baseline in number of edges")
                plot_xticks = np.arange(0, len(allA_C_dict[animal]['trials'][0]), step=60)
                plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
                #plt.xlim(0,180)
                #plt.ylim(-6, 6)
                plt.xlabel("Seconds")
                plt.grid(b=True, which='major', linestyle='-')
                # plot z-scored edge counts
                if animal in allA_M_dict:
                    for trial in allA_M_dict[animal]['trials']:
                        plt.plot(trial, linewidth=1, color=[1.0, 0.0, 0.0, 0.1])
                if animal in allA_C_dict:
                    for trial in allA_C_dict[animal]['trials']:
                        plt.plot(trial, linewidth=1, color=[0.0, 0.0, 1.0, 0.1])
                if animal in allA_M_dict:
                    plt.plot(misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
                if animal in allA_C_dict:
                    plt.plot(catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
                # plot events
                ymin, ymax = plt.ylim()
                plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
                plt.text(baseline_len, ymax-ymax/10, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
                plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
                plt.text(TGB_bucket, ymax-ymax/20, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
                plt.legend(loc='upper left')
                # save fig
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
        except Exception:
            plt.close()
            print("{a} did not make any catches and/or misses during {p} prey movement".format(a=animal,p=prey_type_str))

def check_violations_sigBounds(shuffDiffMeansTraces, sig_upperBound, sig_lowerBound):
    outOfBounds_upper = 0
    outOfBounds_lower = 0
    for trial in shuffDiffMeansTraces:
        num_crossings_UB = 0
        num_crossings_LB = 0
        for x in range(len(trial)):
            if trial[x]>sig_upperBound[x]:
                num_crossings_UB += 1
            if trial[x]<sig_lowerBound[x]:
                num_crossings_LB += 1
        if num_crossings_UB>0:
            outOfBounds_upper += 1
        if num_crossings_LB>0:
            outOfBounds_lower += 1
    return outOfBounds_upper, outOfBounds_lower

def gen_shuffled_traces(Group1, Group2, N_Shuffles, Group1_N, Group2_N):
    # Shuffle the dataset and compare means again
    num_of_shuffles = N_Shuffles
    SPerf = np.zeros((num_of_shuffles,1))
    All_Group = np.concatenate([Group1, Group2])
    for shuff in range(num_of_shuffles):
        shuff_response = np.random.permutation(All_Group)
        SPerf[shuff] = np.nanmean(shuff_response[0:len(Group1)]) - np.nanmean(shuff_response[len(Group1):])
    # sigma
    return SPerf

def find_bounds_for_sig(shuffle_test_results_dict, UpBound, LowBound):
    upperbound = []
    lowerbound = []
    for timebin in sorted(shuffle_test_results_dict.keys()):
        upperbound.append(np.percentile(shuffle_test_results_dict[timebin]['SPerf'], UpBound))
        lowerbound.append(np.percentile(shuffle_test_results_dict[timebin]['SPerf'], LowBound))
    return upperbound, lowerbound

def plot_allA_Zscored_ShuffledDiffMeans(analysis_type_str, preprocess_str, metric_str, prey_type_str, catches_dict, misses_dict, sigUB, sigLB, sigUB_corrected, sigLB_corrected, shuffDiff, firstSigTB, TGB_bucket, baseline_len, plots_dir, todays_dt): 
    img_type = ['.png', '.pdf']
    ### POOL ACROSS ANIMALS ### 
    allA_C = []
    allA_C_N = 0
    allA_M = []
    allA_M_N = 0
    for animal in catches_dict:
        thisA_C_N = len(catches_dict[animal])
        if thisA_C_N != 0:
            allA_C_N = allA_C_N + thisA_C_N
            for trial in catches_dict[animal]:
                allA_C.append(trial)
    for animal in misses_dict:
        thisA_M_N = len(misses_dict[animal])
        if thisA_M_N != 0:
            allA_M_N = allA_M_N + thisA_M_N
            for trial in misses_dict[animal]:
                allA_M.append(trial)
    allA_C_mean = np.nanmean(allA_C, axis=0)
    allA_C_std = np.nanstd(allA_C, axis=0, ddof=1)
    allA_M_mean = np.nanmean(allA_M, axis=0)
    allA_M_std = np.nanstd(allA_M, axis=0, ddof=1)
    ObservedDiff = allA_C_mean - allA_M_mean
    # set fig path and title
    figure_name = analysis_type_str +'_'+ preprocess_str +'_'+ prey_type_str + 'Trials_AllAnimals_' + todays_dt + img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Z-scored mean change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, Pooled across all animals\n Number of catches: {Nc}, Number of misses: {Nm}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, a=animal, Nc=str(allA_C_N), Nm=str(allA_M_N))
    # draw fig
    plt.figure(figsize=(16,16), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    # subplot: real data and std 
    plt.subplot(2,1,1)
    plt.title('Observed data', fontsize=10, color='grey', style='italic')
    plt.ylabel("Z-scored change from baseline in number of edges")
    plot_xticks = np.arange(0, len(allA_C_mean), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.ylim(-1.5,3.0)
    #plt.xlim(0,180)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    # set colors
    color_meanM = [1.0, 0.0, 0.0, 0.8]
    color_stdM = [0.9, 0.0, 0.0, 0.1]
    color_meanC = [0.0, 0.0, 1.0, 0.8]
    color_stdC = [0.0, 0.0, 0.9, 0.1]
    color_pointwiseP005 = [0.0, 0.5, 0.0, 1.0]
    color_globalP005 = [1.0, 0.65, 0.0, 1.0]
    color_obsDiffMeans = [0.0, 0.0, 0.0, 1.0]
    color_shuffDiffMeans = [0.467, 0.537, 0.6, 1.0]
    # plot mean of catches and misses
    x_tbs = range(360)
    UpperBound_M = allA_M_mean + allA_M_std
    LowerBound_M = allA_M_mean - allA_M_std
    UpperBound_C = allA_C_mean + allA_C_std
    LowerBound_C = allA_C_mean - allA_C_std
    plt.plot(allA_M_mean, linewidth=2, color=color_meanM, label='Miss')
    plt.fill_between(x_tbs, UpperBound_M, LowerBound_M, color=color_stdM)
    plt.plot(allA_C_mean, linewidth=2, color=color_meanC, label='Catch')
    plt.fill_between(x_tbs, UpperBound_C, LowerBound_C, color=color_stdC)
    # label events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax-0.75), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-0.75, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-0.25, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    #subplot: difference of observed means vs shuffled diff of means
    plt.subplot(2,1,2)
    plt.title('Significance of the Difference of means (catch vs miss), Number of shuffles = 20000', fontsize=10, color='grey', style='italic')
    plt.ylabel("Difference of z-scored means in number of edges")
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.ylim(-1.5,3.0)
    #plt.xlim(0,180)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    # plot pointwise p<0.05
    plt.plot(sigUB, linestyle='--', color=color_pointwiseP005, label='Pointwise p<0.05')
    plt.plot(sigLB, linestyle='--', color=color_pointwiseP005)
    # plot corrected (global) p<0.05
    plt.plot(sigUB_corrected, linestyle='--', color=color_globalP005, label='Global p<0.05')
    plt.plot(sigLB_corrected, linestyle='--', color=color_globalP005)
    # plot shuffled diff of means
    plt.plot(shuffDiff, linewidth=1.5, linestyle='-', color=color_shuffDiffMeans, label='Shuffled diff of means')
    # plot real diff of means
    plt.plot(ObservedDiff, linewidth=2, linestyle='-', color=color_obsDiffMeans, label='Observed diff of means')
    # plot significant time bins as shaded region
    sig_x = range(firstSigTB, 360)
    plt.fill_between(sig_x, ObservedDiff[firstSigTB:], sigUB[firstSigTB:], color='cyan', alpha=0.3)
    # label events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax-0.75), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-0.75, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-0.25, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.plot((firstSigTB, firstSigTB), (ymin, ymax-0.75), 'c--', linewidth=1)
    plt.text(firstSigTB, ymax-0.75, "Difference between \n catches and misses becomes \nsignificant at {s:.2f} seconds".format(s=firstSigTB/60), fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='cyan', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    # save and show fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

### BEGIN ANALYSIS ###
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()
# List relevant data locations: these are for KAMPFF-LAB
#root_folder = r"C:\Users\Kampff_Lab\Documents\Github\CuttleShuttle-Analysis\Workflows"
#canny_counts_folder = r"C:\Users\Kampff_Lab\Documents\Github\CuttleShuttle-Analysis\Workflows\CannyCount_csv_smallCrop_Canny2000-7500"
#plots_folder = r"C:\Users\Kampff_Lab\Documents\Github\CuttleShuttle-Analysis\Workflows\plots"

# List relevant data locations: these are for taunsquared
root_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows"
old_canny_counts_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\CannyCount_csv_smallCrop_Canny2000-7500"
canny_counts_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\CannyCount_20191025"
sobel_score_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\SobelScore_20191025"
pixel_sum_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\PixelSum_20191025"
plots_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\plots"

# in canny_counts_folder, list all csv files for TGB moments ("Tentacles Go Ballistic")
TGB_all = glob.glob(canny_counts_folder + os.sep + "*.csv")

# categorize tentacle shots according to prey movement
TGB_natural = []
TGB_patterned = []
TGB_causal = []
TGB_daily = {}
for TGB_file in TGB_all: 
    csv_name = TGB_file.split(os.sep)[-1]
    trial_date = csv_name.split('_')[2]
    sorted_by_session = TGB_daily.setdefault(trial_date,[]).append(TGB_file)
    trial_datetime = datetime.datetime.strptime(trial_date, '%Y-%m-%d')
    if trial_datetime < datetime.datetime(2014, 9, 13, 0, 0):
        TGB_natural.append(TGB_file)
    elif trial_datetime > datetime.datetime(2014, 10, 18, 0, 0):
        TGB_causal.append(TGB_file)
    else: 
        TGB_patterned.append(TGB_file)

# organize canny count data
# categorize daily sessions by animal
all_TS_daily = {}
all_catches_daily = {}
all_misses_daily = {}
for session_date in TGB_daily:
    all_TS_daily[session_date] = categorize_by_animal(TGB_daily[session_date])
    all_catches_daily[session_date], all_misses_daily[session_date] = categorize_by_animal_catchVmiss(TGB_daily[session_date])
# collect all canny counts and categorize by animal
all_TS = categorize_by_animal(TGB_all)
# collect all canny counts and categorize by animal and type (catch vs miss)
all_catches, all_misses = categorize_by_animal_catchVmiss(TGB_all)
# organize by prey type
all_raw = [all_catches, all_misses]
# time bin for moment tentacles go ballistic
TGB_bucket_raw = 180

########################################################
### ------ DATA NORMALIZATION/STANDARDIZATION ------ ###
########################################################

# BASELINE SUBTRACTION 
baseline_buckets = 150
# sav-gol filter and baseline subtract, all TS
savgol_window = 15
dailyTS_filtBaseSub = {}
for session_date in all_TS_daily:
    dailyTS_filtBaseSub[session_date] = filtered_basesub_count(all_TS_daily[session_date], 'all', baseline_buckets, savgol_window)
allTS_filtBaseSub = filtered_basesub_count(all_TS, 'all', baseline_buckets, savgol_window)
# sav-gol filter and baseline subtract, catches versus misses
dailyCatches_filtBaseSub = {}
dailyMisses_filtBaseSub = {}
for session_date in all_catches_daily:
    dailyCatches_filtBaseSub[session_date] = filtered_basesub_count(all_catches_daily[session_date], 'all', baseline_buckets, savgol_window)
for session_date in all_misses_daily:
    dailyMisses_filtBaseSub[session_date] = filtered_basesub_count(all_misses_daily[session_date], 'all', baseline_buckets, savgol_window)
allCatches_filtBaseSub = filtered_basesub_count(all_catches, 'all', baseline_buckets, savgol_window)
allMisses_filtBaseSub = filtered_basesub_count(all_misses, 'all', baseline_buckets, savgol_window)
# zscore each animal so that I can pool all trials into a "superanimal"
allTS_filtBaseSub_Zscored = zScored_count('timebin', allTS_filtBaseSub, allTS_filtBaseSub)
allCatches_filtBaseSub_Zscored_TB = zScored_count('timebin', allCatches_filtBaseSub, allTS_filtBaseSub)
allMisses_filtBaseSub_Zscored_TB = zScored_count('timebin', allMisses_filtBaseSub, allTS_filtBaseSub)
allTS_filtBaseSub_Zscored_TB_Sess = zScored_count('session', allTS_filtBaseSub, allTS_filtBaseSub)
allCatches_filtBaseSub_Zscored_Sess = zScored_count('session', allCatches_filtBaseSub, allTS_filtBaseSub)
allMisses_filtBaseSub_Zscored_Sess = zScored_count('session', allMisses_filtBaseSub, allTS_filtBaseSub)
# zscore daily sessions for each animal to characterize session dynamics
dailyTS_filtBaseSub_Zscored_Sess = {}
for session_date in dailyTS_filtBaseSub:
    dailyTS_filtBaseSub_Zscored_Sess[session_date] = zScored_count('session', dailyTS_filtBaseSub[session_date], dailyTS_filtBaseSub[session_date])
dailyCatches_filtBaseSub_Zscored_Sess = {}
dailyMisses_filtBaseSub_Zscored_Sess = {}
for session_date in dailyCatches_filtBaseSub:
    dailyCatches_filtBaseSub_Zscored_Sess[session_date] = zScored_count('session', dailyCatches_filtBaseSub[session_date], dailyTS_filtBaseSub[session_date])
for session_date in dailyMisses_filtBaseSub:
    dailyMisses_filtBaseSub_Zscored_Sess[session_date] = zScored_count('session', dailyMisses_filtBaseSub[session_date], dailyTS_filtBaseSub[session_date])

#######################################################
### ------------ PLOT THE ZSCORED DATA ------------ ###
#######################################################

## individual animals
plot_indiv_animals('CannyEdgeDetector', 'Zscored_TB_SavGol_BaseSub', 'edge counts', 'all', allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)
plot_indiv_animals('CannyEdgeDetector', 'Zscored_Sess_SavGol_BaseSub', 'edge counts', 'all', allCatches_filtBaseSub_Zscored_Sess, allMisses_filtBaseSub_Zscored_Sess, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

# sanity check
for session_date in dailyTS_filtBaseSub:
    plot_indiv_animals('CannyEdgeDetector', 'SavGol_BaseSub', 'edge counts', 'all '+session_date, dailyCatches_filtBaseSub[session_date], dailyMisses_filtBaseSub[session_date], TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)
    plot_indiv_animals('CannyEdgeDetector', 'Zscored_Sess_SavGol_Basesub', 'edge counts', 'all '+session_date, dailyCatches_filtBaseSub_Zscored_Sess[session_date], dailyMisses_filtBaseSub_Zscored_Sess[session_date], TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

########################################################
### -------- SHUFFLE TESTS FOR SIGNIFICANCE -------- ###
########################################################
No_of_Shuffles = 20000

### POOL ACROSS ALL ANIMALS, zscored, full trial
allA_ZfullTrial_C, allA_ZfullTrial_C_N, allA_ZfullTrial_M, allA_ZfullTrial_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 0, -1, "all")
# all animals full trial shuffle test
allA_ZfullTrial_SPerf, allA_ZfullTrial_pval, allA_ZfullTrial_mean = shuffle_test(allA_ZfullTrial_C, allA_ZfullTrial_M, No_of_Shuffles, "AllCatches-Zscored-fullTrial", "AllMisses-Zscored-fullTrial", allA_ZfullTrial_C_N, allA_ZfullTrial_M_N, False, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, before and after TGB
allA_ZpreTGB_C, allA_ZpreTGB_C_N, allA_ZpreTGB_M, allA_ZpreTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 0, TGB_bucket_raw-1, "all")
allA_ZpostTGB_C, allA_ZpostTGB_C_N, allA_ZpostTGB_M, allA_ZpostTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, TGB_bucket_raw, -1, "all")
# all animals preTGB shuffle test
allA_ZpreTGB_SPerf, allA_ZpreTGB_pval, allA_ZpreTGB_mean = shuffle_test(allA_ZpreTGB_C, allA_ZpreTGB_M, No_of_Shuffles, "AllCatch-Zscored-preTGB", "AllMiss-Zscored-preTGB", allA_ZpreTGB_C_N, allA_ZpreTGB_M_N, False, plots_folder, todays_datetime)
# all animals postTGB shuffle test
allA_ZpostTGB_SPerf, allA_ZpostTGB_pval, allA_ZpostTGB_mean = shuffle_test(allA_ZpostTGB_C, allA_ZpostTGB_M, No_of_Shuffles, "AllCatch-Zscored-postTGB", "AllMiss-Zscored-postTGB", allA_ZpostTGB_C_N, allA_ZpostTGB_M_N, False, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, baseline period
allA_Zbaseline_C, allA_Zbaseline_C_N, allA_Zbaseline_M, allA_Zbaseline_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 0, baseline_buckets, "all")
# all animals baseline shuffle test
allA_Zbaseline_SPerf, allA_Zbaseline_pval, allA_Zbaseline_mean = shuffle_test(allA_Zbaseline_C, allA_Zbaseline_M, No_of_Shuffles, "AllCatch-Zscored-baseline", "AllMiss-Zscored-baseline", allA_Zbaseline_C_N, allA_Zbaseline_M_N, False, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, 0.5 seconds before TGB
# zscored by timebin
allA_ZhalfSecPreTGB_C, allA_ZhalfSecPreTGB_C_N, allA_ZhalfSecPreTGB_M, allA_ZhalfSecPreTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, baseline_buckets, TGB_bucket_raw-1, "all")
# all animals baseline shuffle test
allA_ZhalfSecPreTGB_SPerf, allA_ZhalfSecPreTGB_pval, allA_ZhalfSecPreTGB_mean = shuffle_test(allA_ZhalfSecPreTGB_C, allA_ZhalfSecPreTGB_M, No_of_Shuffles, "AllCatch-Zscored-halfSecPreTGB", "AllMiss-Zscored-halfSecPreTGB", allA_ZhalfSecPreTGB_C_N, allA_ZhalfSecPreTGB_M_N, False, plots_folder, todays_datetime)
### individual animals, 0.5 seconds before TGB (show trends in individual animals to double check the pooled shuffle test)
indivA_N_shuffles = 1000
byA_ZhalfSecPreTGB_C, byA_ZhalfSecPreTGB_M = pool_timebins_byAnimal(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, baseline_buckets, TGB_bucket_raw-1)
byA_ZhalfSecPreTGB_SPerf = {}
byA_ZhalfSecPreTGB_pval = {}
byA_ZhalfSecPreTGB_mean = {}
for animal in byA_ZhalfSecPreTGB_C:
    thisA_ZhalfSecPreTGB_SPerf, thisA_ZhalfSecPreTGB_pval, thisA_ZhalfSecPreTGB_mean = shuffle_test(byA_ZhalfSecPreTGB_C[animal], byA_ZhalfSecPreTGB_M[animal], indivA_N_shuffles, animal+"-Zscored-halfSecPreTGB", animal+"-Zscored-halfSecPreTGB", len(allCatches_filtBaseSub_Zscored_TB[animal]), len(allMisses_filtBaseSub_Zscored_TB[animal]), False, plots_folder, todays_datetime)
    byA_ZhalfSecPreTGB_SPerf[animal] = thisA_ZhalfSecPreTGB_SPerf
    byA_ZhalfSecPreTGB_pval[animal] = thisA_ZhalfSecPreTGB_pval
    byA_ZhalfSecPreTGB_mean[animal] = thisA_ZhalfSecPreTGB_mean

### POOL ACROSS ALL ANIMALS, from 3.4 seconds after TGB to end
allA_Ztb205toEnd_C, allA_Ztb205toEnd_C_N, allA_Ztb205toEnd_M, allA_Ztb205toEnd_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 205, -1, "all")
# all animals baseline shuffle test
allA_Ztb205toEnd_SPerf, allA_Ztb205toEnd_pval, allA_Ztb205toEnd_mean = shuffle_test(allA_Ztb205toEnd_C, allA_Ztb205toEnd_M, No_of_Shuffles, "AllCatch-Zscored-tb205toEnd", "AllMiss-Zscored-tb205toEnd", allA_Ztb205toEnd_C_N, allA_Ztb205toEnd_M_N, False, plots_folder, todays_datetime)
### individual animals, from 3.4 seconds after TGB to end (show trends in individual animals to double check the pooled shuffle test)
byA_ZpostTB205_C, byA_ZpostTB205_M = pool_timebins_byAnimal(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 205, -1)
byA_ZpostTB205_SPerf = {}
byA_ZpostTB205_pval = {}
byA_ZpostTB205_mean = {}
for animal in byA_ZpostTB205_C:
    thisA_ZpostTB205_SPerf, thisA_ZpostTB205_pval, thisA_ZpostTB205_mean = shuffle_test(byA_ZpostTB205_C[animal], byA_ZpostTB205_M[animal], indivA_N_shuffles, animal+"-Catches-Zscored-tb205toEnd", animal+"-Misses-Zscored-tb205toEnd", len(allCatches_filtBaseSub_Zscored_TB[animal]), len(allMisses_filtBaseSub_Zscored_TB[animal]), False, plots_folder, todays_datetime)
    byA_ZpostTB205_SPerf[animal] = thisA_ZpostTB205_SPerf
    byA_ZpostTB205_pval[animal] = thisA_ZpostTB205_pval
    byA_ZpostTB205_mean[animal] = thisA_ZpostTB205_mean

### POOL ACROSS ALL ANIMALS, make a shuffle test of every time bin
# zscored by timebin
allA_Z_byTB_C, allA_Z_byTB_C_N, allA_Z_byTB_M, allA_Z_byTB_M_N = pool_acrossA_keepTemporalStructure(allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, 0, -1, "all")
ZedgeScores_byTB = {}
for timebin in range(360):
    # collect all edge scores for each time bin
    ZedgeScores_byTB[timebin] = {'catch':[], 'miss':[], 'SPerf': None, 'pval': None, 'mean': None}
    for trial in allA_Z_byTB_C:
        ZedgeScores_byTB[timebin]['catch'].append(trial[timebin])
    for trial in allA_Z_byTB_M:
        ZedgeScores_byTB[timebin]['miss'].append(trial[timebin])
    # shuffle test each time bin
    ZedgeScores_byTB[timebin]['SPerf'], ZedgeScores_byTB[timebin]['pval'], ZedgeScores_byTB[timebin]['mean'] = shuffle_test(ZedgeScores_byTB[timebin]['catch'], ZedgeScores_byTB[timebin]['miss'], No_of_Shuffles, 'AllCatches-Zscored-TB'+str(timebin), 'AllMisses-Zscored-TB'+str(timebin), allA_Z_byTB_C_N, allA_Z_byTB_M_N, False, plots_folder, todays_datetime)
# zscored by entire dataset
allA_ZSess_byTB_C, allA_ZSess_byTB_C_N, allA_ZSess_byTB_M, allA_ZSess_byTB_M_N = pool_acrossA_keepTemporalStructure(allCatches_filtBaseSub_Zscored_Sess, allMisses_filtBaseSub_Zscored_Sess, 0, -1, "all")
ZSessEdgeScores_byTB = {}
for timebin in range(360):
    # collect all edge scores for each time bin
    ZSessEdgeScores_byTB[timebin] = {'catch':[], 'miss':[], 'SPerf': None, 'pval': None, 'mean': None}
    for trial in allA_ZSess_byTB_C:
        ZSessEdgeScores_byTB[timebin]['catch'].append(trial[timebin])
    for trial in allA_ZSess_byTB_M:
        ZSessEdgeScores_byTB[timebin]['miss'].append(trial[timebin])
    # shuffle test each time bin
    ZSessEdgeScores_byTB[timebin]['SPerf'], ZSessEdgeScores_byTB[timebin]['pval'], ZSessEdgeScores_byTB[timebin]['mean'] = shuffle_test(ZSessEdgeScores_byTB[timebin]['catch'], ZSessEdgeScores_byTB[timebin]['miss'], No_of_Shuffles, 'AllCatches-ZscoredSess-TB'+str(timebin), 'AllMisses-ZscoredSess-TB'+str(timebin), allA_ZSess_byTB_C_N, allA_ZSess_byTB_M_N, True, plots_folder, todays_datetime)

#######################################################
### -- CALCULATE UPPER & LOWER BOUNDS FOR P<0.05 -- ###
#######################################################

# pointwise p<0.05 bounds
UB_pointwise = 97.5
LB_pointwise = 2.5
pw005sig_UB, pw005sig_LB = find_bounds_for_sig(ZedgeScores_byTB, UB_pointwise, LB_pointwise)
pw005sig_Zsess_UB, pw005sig_Zsess_LB = find_bounds_for_sig(ZSessEdgeScores_byTB, UB_pointwise, LB_pointwise)

# collect shuffled mean of each time bin
shuff_DiffMeans = []
for timebin in sorted(ZedgeScores_byTB.keys()):
    shuff_DiffMeans.append(ZedgeScores_byTB[timebin]['mean'])
shuff_DiffMeans = np.array(shuff_DiffMeans)
#
shuff_ZSess_DiffMeans = []
for timebin in sorted(ZSessEdgeScores_byTB.keys()):
    shuff_ZSess_DiffMeans.append(ZSessEdgeScores_byTB[timebin]['mean'])
shuff_ZSess_DiffMeans = np.array(shuff_ZSess_DiffMeans)

# calculate real difference of mean catch and mean miss
allA_allC_Z = []
allA_allM_Z = []
for animal in allCatches_filtBaseSub_Zscored_TB:
    for trial in allCatches_filtBaseSub_Zscored_TB[animal]:
        allA_allC_Z.append(trial)
    for trial in allMisses_filtBaseSub_Zscored_TB[animal]:
        allA_allM_Z.append(trial)
allA_allC_Z_mean = np.nanmean(allA_allC_Z, axis=0)
allA_allM_Z_mean = np.nanmean(allA_allM_Z, axis=0)
Observed_DiffMeans = allA_allC_Z_mean - allA_allM_Z_mean
#
allA_allC_ZSess = []
allA_allM_ZSess = []
for animal in allCatches_filtBaseSub_Zscored_Sess:
    for trial in allCatches_filtBaseSub_Zscored_Sess[animal]:
        allA_allC_ZSess.append(trial)
    for trial in allMisses_filtBaseSub_Zscored_Sess[animal]:
        allA_allM_ZSess.append(trial)
allA_allC_ZSess_mean = np.nanmean(allA_allC_ZSess, axis=0)
allA_allM_ZSess_mean = np.nanmean(allA_allM_ZSess, axis=0)
Observed_DiffMeans_ZSess = allA_allC_ZSess_mean - allA_allM_ZSess_mean

# generate random traces to correct threshold for p<0.05
No_of_random_traces = 1000
# zscored by timebin
shuffledDiffMeans_byTB = {}
for timebin in ZedgeScores_byTB:
    shuffledDiffMeans_byTB[timebin] = gen_shuffled_traces(ZedgeScores_byTB[timebin]['catch'], ZedgeScores_byTB[timebin]['miss'], No_of_random_traces, len(ZedgeScores_byTB[timebin]['catch']), len(ZedgeScores_byTB[timebin]['miss']))
# convert to arrays for plotting
shuffMeans_traces = []
shuffMeans_traces_N = len(shuffledDiffMeans_byTB[0])
for st in range(shuffMeans_traces_N):
    this_trace = []
    for timebin in shuffledDiffMeans_byTB:
        this_trace.append(shuffledDiffMeans_byTB[timebin][st][0])
    shuffMeans_traces.append(this_trace)
shuffMeans_traces = np.array(shuffMeans_traces)
# zscored by entire dataset
shuffledDiffMeans_ZSess_byTB = {}
for timebin in ZSessEdgeScores_byTB:
    shuffledDiffMeans_ZSess_byTB[timebin] = gen_shuffled_traces(ZSessEdgeScores_byTB[timebin]['catch'], ZSessEdgeScores_byTB[timebin]['miss'], No_of_random_traces, len(ZSessEdgeScores_byTB[timebin]['catch']), len(ZSessEdgeScores_byTB[timebin]['miss']))
# convert to arrays for plotting
shuffMeans_ZSess_traces = []
shuffMeans_ZSess_traces_N = len(shuffledDiffMeans_ZSess_byTB[0])
for st in range(shuffMeans_ZSess_traces_N):
    this_trace = []
    for timebin in shuffledDiffMeans_ZSess_byTB:
        this_trace.append(shuffledDiffMeans_ZSess_byTB[timebin][st][0])
    shuffMeans_ZSess_traces.append(this_trace)
shuffMeans_ZSess_traces = np.array(shuffMeans_ZSess_traces)

# correct the p<0.05 bounds
UB_corrected = 99.994
LB_corrected = 0.006
global005sig_UB, global005sig_LB = find_bounds_for_sig(ZedgeScores_byTB, UB_corrected, LB_corrected)
global005sig_ZSess_UB, global005sig_ZSess_LB = find_bounds_for_sig(ZSessEdgeScores_byTB, UB_corrected, LB_corrected)

# check how many of these random traces violate the p<0.05 generated by timebin-wise shuffle test
N_violations_UBcorrected, N_violations_LBcorrected = check_violations_sigBounds(shuffMeans_traces, global005sig_UB, global005sig_LB)
N_violations_ZSess_UBcorrected, N_violations_ZSess_LBcorrected = check_violations_sigBounds(shuffMeans_ZSess_traces, global005sig_ZSess_UB, global005sig_ZSess_LB)

# find where observed data crosses corrected bounds for first time
for tb in range(len(Observed_DiffMeans)):
    if Observed_DiffMeans[tb]>pw005sig_UB[tb]:
        firstTB_P005sig = tb
        break
# 
for tb in range(len(Observed_DiffMeans_ZSess)):
    if Observed_DiffMeans_ZSess[tb]>pw005sig_Zsess_UB[tb]:
        firstTB_ZSess_P005sig = tb
        break

# visualize
for shuff_trace in shuffMeans_traces:
    plt.plot(shuff_trace, alpha=0.1)
plt.plot(pw005sig_UB, 'g--')
plt.plot(pw005sig_LB, 'g--')
plt.plot(global005sig_UB, 'm--')
plt.plot(global005sig_LB, 'm--')
plt.plot(shuff_DiffMeans, 'b--')
plt.plot(Observed_DiffMeans, 'k-')
plt.show()
#
for shuff_trace in shuffMeans_ZSess_traces:
    plt.plot(shuff_trace, alpha=0.1)
plt.plot(pw005sig_Zsess_UB, 'g--')
plt.plot(pw005sig_Zsess_LB, 'g--')
plt.plot(global005sig_ZSess_UB, 'm--')
plt.plot(global005sig_ZSess_LB, 'm--')
plt.plot(shuff_ZSess_DiffMeans, 'b--')
plt.plot(Observed_DiffMeans_ZSess, 'k-')
plt.show()

#######################################################
### ------------ PLOT THE SHUFFLE DATA ------------ ###
#######################################################

### POOL ACROSS ANIMALS
plot_allA_Zscored_ShuffledDiffMeans('CannyEdgeDetector', 'Zscored_SavGol_BaseSub', 'edge counts', 'all', allCatches_filtBaseSub_Zscored_TB, allMisses_filtBaseSub_Zscored_TB, pw005sig_UB, pw005sig_LB, global005sig_UB, global005sig_LB, shuff_DiffMeans, firstTB_P005sig, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)
plot_allA_Zscored_ShuffledDiffMeans('CannyEdgeDetector', 'ZscoredSess_SavGol_BaseSub', 'edge counts', 'all', allCatches_filtBaseSub_Zscored_Sess, allMisses_filtBaseSub_Zscored_Sess, pw005sig_Zsess_UB, pw005sig_Zsess_LB, global005sig_ZSess_UB, global005sig_ZSess_LB, shuff_ZSess_DiffMeans, firstTB_ZSess_P005sig, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

##############################################################################
### -- GET ERROR BAR FOR WHEN DIFF B/T CATCH V MISS BECOMES SIGNIFICANT -- ###
##############################################################################

ZSessEdgeScores_byTB_ErrOfSigTB = {}
pw005sig_Zsess_ErrOfSigTB = {}
shuff_ZSess_DiffMeans_ErrOfSigTB = {}
shuffledDiffMeans_ZSess_byTB_ErrOfSigTB = {}
shuffMeans_ZSess_traces_ErrOfSigTB = {}
UB_corrected_ErrOfSigTB = {}
LB_corrected_ErrOfSigTB = {}
firstTB_ZSess_P005sig_ErrOfSigTB = {}
No_of_repetitions = 100
for rep in No_of_repetitions:
    # shuffle test
    print('Running shuffle test, rep = {r}...'.format(r=rep))
    ZSessEdgeScores_byTB_ErrOfSigTB[rep] = {}
    pw005sig_Zsess_ErrOfSigTB[rep] = {'upper bound': None, 'lower bound': None}
    for timebin in range(360):
        # collect all edge scores for each time bin
        ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin] = {'catch':[], 'miss':[], 'SPerf': None, 'pval': None, 'mean': None}
        for trial in allA_ZSess_byTB_C:
            ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['catch'].append(trial[timebin])
        for trial in allA_ZSess_byTB_M:
            ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['miss'].append(trial[timebin])
        # shuffle test each time bin
        ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['SPerf'], ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['pval'], ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['mean'] = shuffle_test(ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['catch'], ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['miss'], No_of_Shuffles, 'AllCatches-ZscoredSess-TB'+str(timebin), 'AllMisses-ZscoredSess-TB'+str(timebin), allA_ZSess_byTB_C_N, allA_ZSess_byTB_M_N, False, plots_folder, todays_datetime)
    # collect shuffled mean of each time bin
    shuff_ZSess_DiffMeans_ErrOfSigTB[rep] = []
    for timebin in sorted(ZSessEdgeScores_byTB_ErrOfSigTB[rep].keys()):
        shuff_ZSess_DiffMeans_ErrOfSigTB[rep].append(ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['mean'])
    shuff_ZSess_DiffMeans_ErrOfSigTB[rep] = np.array(shuff_ZSess_DiffMeans_ErrOfSigTB[rep])
    # pointwise p<0.05
    pw005sig_Zsess_ErrOfSigTB[rep]['upper bound'], pw005sig_Zsess_ErrOfSigTB[rep]['lower bound'] = find_bounds_for_sig(ZSessEdgeScores_byTB_ErrOfSigTB[rep], UB_pointwise, LB_pointwise)
    # generate random traces to correct threshold for p<0.05
    print('Generating 1000 random traces, rep = {r}...'.format(r=rep))
    shuffledDiffMeans_ZSess_byTB_ErrOfSigTB[rep] = {}
    for timebin in ZSessEdgeScores_byTB_ErrOfSigTB[rep]:
        shuffledDiffMeans_ZSess_byTB_ErrOfSigTB[rep][timebin] = gen_shuffled_traces(ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['catch'], ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['miss'], No_of_random_traces, len(ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['catch']), len(ZSessEdgeScores_byTB_ErrOfSigTB[rep][timebin]['miss']))
    # convert to arrays for plotting
    shuffMeans_ZSess_traces_ErrOfSigTB[rep] = []
    shuffMeans_ZSess_traces_N_ErrOfSigTB = len(shuffledDiffMeans_ZSess_byTB_ErrOfSigTB[rep][0])
    for st in range(shuffMeans_ZSess_traces_N_ErrOfSigTB):
        this_trace = []
        for timebin in shuffledDiffMeans_ZSess_byTB_ErrOfSigTB[rep]:
            this_trace.append(shuffledDiffMeans_ZSess_byTB_ErrOfSigTB[rep][timebin][st][0])
        shuffMeans_ZSess_traces_ErrOfSigTB[rep].append(this_trace)
    shuffMeans_ZSess_traces_ErrOfSigTB[rep] = np.array(shuffMeans_ZSess_traces_ErrOfSigTB[rep])
    # correct the p<0.05 bounds
    print('Correcting p<0.05 bounds, rep = {r}...'.format(r=rep))
    UB_corrected_ErrOfSigTB[rep] = UB_pointwise
    LB_corrected_ErrOfSigTB[rep] = LB_pointwise
    N_violations_ZSess_UBcorrected_ErrOfSigTB = 1000
    N_violations_ZSess_LBcorrected_ErrOfSigTB = 1000
    while N_violations_ZSess_UBcorrected_ErrOfSigTB>50 or N_violations_ZSess_LBcorrected_ErrOfSigTB>50:
        global005sig_ZSess_UB_ErrOfSigTB[rep], global005sig_ZSess_LB_ErrOfSigTB[rep] = find_bounds_for_sig(ZSessEdgeScores_byTB_ErrOfSigTB[rep], UB_corrected_ErrOfSigTB[rep], LB_corrected_ErrOfSigTB[rep])
        # check how many of these random traces violate the p<0.05 generated by timebin-wise shuffle test
        N_violations_ZSess_UBcorrected_ErrOfSigTB, N_violations_ZSess_LBcorrected_ErrOfSigTB = check_violations_sigBounds(shuffMeans_ZSess_traces_ErrOfSigTB[rep], global005sig_ZSess_UB_ErrOfSigTB[rep], global005sig_ZSess_LB_ErrOfSigTB[rep])
        if N_violations_ZSess_UBcorrected_ErrOfSigTB>50 or N_violations_ZSess_LBcorrected_ErrOfSigTB>50:
            UB_corrected_ErrOfSigTB[rep] = UB_corrected_ErrOfSigTB[rep] + 0.001
            LB_corrected_ErrOfSigTB[rep] = LB_corrected_ErrOfSigTB[rep] - 0.001
    # find where observed data crosses corrected bounds for first time
    for tb in range(len(Observed_DiffMeans_ZSess)):
    if Observed_DiffMeans_ZSess[tb]>pw005sig_Zsess_ErrOfSigTB[rep]['upper bound'][tb]:
        firstTB_ZSess_P005sig_ErrOfSigTB[rep] = tb
        break

### TO DO
# calculate the average time post-TGB when tentacles hit target and when tentacles return to mouth and add to plots


## FIN