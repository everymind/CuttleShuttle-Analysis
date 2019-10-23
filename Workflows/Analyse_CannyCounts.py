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
from fitter import Fitter
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
            basesub_filtered_mean = np.nanmean(all_filtered_basesub_trials, axis=0)
            basesub_filtered_std = np.nanstd(all_filtered_basesub_trials, axis=0, ddof=1)
            basesub_filtered_TS[animal]['trials'] = all_filtered_basesub_trials
            basesub_filtered_TS[animal]['mean'] = basesub_filtered_mean
            basesub_filtered_TS[animal]['std'] = basesub_filtered_std
        except Exception:
            print("{a} made no tentacle shots during {p} prey movement type".format(a=animal, p=prey_type))
    return basesub_filtered_TS

def zScored_count(dict_to_Zscore, dict_for_mean_std):
    zScored_dict = {}
    for animal in dict_to_Zscore:
        zScored_dict[animal] = []
        for trial in dict_to_Zscore[animal]['trials']:
            trial_array = np.array(trial)
            trial_zscored = (trial_array - dict_for_mean_std[animal]['mean'])/dict_for_mean_std[animal]['std']
            zScored_dict[animal].append(trial_zscored)
    return zScored_dict

def shuffle_test(Group1, Group2, N_Shuffles, Group1_str, Group2_str, Group1_N, Group2_N, plot_on, plots_dir, todays_dt):
    # Observed performance
    OPerf = np.mean(Group1) - np.mean(Group2)
    # Shuffle the dataset and compare means again
    num_of_shuffles = N_Shuffles
    SPerf = np.zeros((num_of_shuffles,1))
    All_Group = np.concatenate([Group1, Group2])
    for shuff in range(num_of_shuffles):
        shuff_response = np.random.permutation(All_Group)
        SPerf[shuff] = np.nanmean(shuff_response[0:len(Group1)]) - np.nanmean(shuff_response[len(Group1):])
    # p-value of shuffle test
    pVal = np.mean(SPerf**2 >= OPerf**2)
    # sigma
    shuffled_mean = np.mean(SPerf)
    sigma_shuff = np.std(SPerf, ddof=1)
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
    return pVal, shuff_025p, shuff_975p, shuffled_mean

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





def prob_of_tb_above_edgecount_thresh(tbs_to_check, threshold, basesubfilt_catches, basesubfilt_misses):
    tb_edgecounts_catches = {}
    tb_edgecounts_misses = {}
    prob_above_thresh_catches = {}
    prob_above_thresh_misses = {}
    for timebin in tbs_to_check:
        tb_edgecounts_catches[timebin] = {}
        tb_edgecounts_misses[timebin] = {}
        prob_above_thresh_catches[timebin] = {}
        prob_above_thresh_misses[timebin] = {}
        for animal in basesubfilt_catches:
            tb_edgecounts_catches[timebin][animal] = []
            tb_edgecounts_misses[timebin][animal] = []
            for trial in basesubfilt_catches[animal]:
                tb_edgecounts_catches[timebin][animal].append(trial[timebin-1])
            for trial in basesubfilt_misses[animal]:
                tb_edgecounts_misses[timebin][animal].append(trial[timebin-1])
        for animal in tb_edgecounts_catches[timebin]:
            plus1mil_catches = [x>threshold for x in tb_edgecounts_catches[timebin][animal]]
            prob_above_thresh_catches[timebin][animal] = sum(plus1mil_catches)/len(plus1mil_catches)
            plus1mil_misses = [x>threshold for x in tb_edgecounts_misses[timebin][animal]]
            prob_above_thresh_misses[timebin][animal] = sum(plus1mil_misses)/len(plus1mil_misses)
    return prob_above_thresh_catches, prob_above_thresh_misses

def plot_indiv_animals_BSF_TP(prey_type_str, threshold_str, catches_basesubfilt, misses_basesubfilt, catches_FBS_mean, misses_FBS_mean, prob_aboveThresh_catch, prob_aboveThresh_miss, TGB_bucket, baseline_len, plots_dir, todays_dt):
    # plot individual animals
    img_type = ['.png', '.pdf']
    for animal in catches_basesubfilt.keys(): 
        try:
            #canny_std_catch = np.nanstd(catches_basesubfilt[animal], axis=0, ddof=1)
            canny_N_catch = len(catches_basesubfilt[animal])
            #canny_std_miss = np.nanstd(misses_basesubfilt[animal], axis=0, ddof=1)
            canny_N_miss = len(misses_basesubfilt[animal])
            catches_mean = catches_FBS_mean[animal]
            misses_mean = misses_FBS_mean[animal]

            figure_name = 'CannyEdgeDetector_BaselineSubtracted_SavGolFiltered_WithThreshProb_'+ prey_type_str + 'Trials_' + animal + "_" + todays_dt + img_type[0]
            figure_path = os.path.join(plots_dir, figure_name)
            figure_title = "Mean change from baseline in number of edges in ROI on cuttlefish mantle during tentacle shots, as detected by Canny Edge Detector \n Individual trials plotted with more transparent traces \n Baseline: mean of edge counts from t=0 to t=" + str(baseline_len/60) + " seconds \n Prey Movement type: " + prey_type_str + ", Animal: " + animal + "\n Number of catches: " + str(canny_N_catch) + ", Number of misses: " + str(canny_N_miss)
            plt.figure(figsize=(16,9), dpi=200)
            plt.suptitle(figure_title, fontsize=12, y=0.99)
            plt.ylabel("Change from baseline in number of edges")
            plot_xticks = np.arange(0, len(catches_FBS_mean[animal]), step=60)
            plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
            plt.ylim(-1500000,4000000)
            #plt.xlim(0,180)
            plt.xlabel("Seconds")
            #plt.xlabel("Frame number, original framerate = 60fps")
            plt.grid(b=True, which='major', linestyle='-')
            ymin, ymax = plt.ylim()

            for trial in misses_basesubfilt[animal]:
                plt.plot(trial, linewidth=1, color=[1.0, 0.0, 0.0, 0.1])
            for trial in catches_basesubfilt[animal]:
                plt.plot(trial, linewidth=1, color=[0.0, 0.0, 1.0, 0.1])
            plt.plot(misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
            #plt.fill_between(range(len(misses_basesubfilt_mean[animal])), misses_mean-canny_std_miss, misses_mean+canny_std_miss, color=[1.0, 0.0, 0.0, 0.1])
            plt.plot(catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
            #plt.fill_between(range(len(catches_basesubfilt_mean[animal])), catches_mean-canny_std_catch, catches_mean+canny_std_catch, color=[0.0, 0.0, 1.0, 0.1])
            for timebin in prob_aboveThresh_miss:
                plt.plot(timebin, misses_mean[timebin], 'ro')
                plt.plot((timebin, timebin), (ymin, ymax), 'k--', linewidth=1)
                plt.text(timebin, misses_mean[timebin]-1000000, '%.1f'%(timebin/60 - 3)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_miss[timebin][animal]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
            for timebin in prob_aboveThresh_catch:
                plt.plot(timebin, catches_mean[timebin], 'bo')
                plt.text(timebin, catches_mean[timebin]+1000000, '%.1f'%(timebin/60 - 3)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_catch[timebin][animal]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.35'))

            plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
            plt.text(baseline_len, ymax-600000, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
            plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
            plt.text(TGB_bucket, ymax-250000, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
            plt.legend(loc='upper left')

            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        except Exception:
            plt.close()
            print("{a} did not make any catches and/or misses during {p} prey movement".format(a=animal,p=prey_type_str))

def mult_to_list(input_list, multiplier):
    if not np.isnan(input_list).any(): 
        return [x*multiplier for x in input_list]

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



prey_type_str = "all"
threshold_str = "1 million"
catches_basesubfilt = all_catches_basesub_filtered
misses_basesubfilt = all_misses_basesub_filtered
catches_FBS_mean = all_catches_basesub_filtered_mean
misses_FBS_mean = all_misses_basesub_filtered_mean
edgeScoresTB_dict = ZedgeScores_byTB
ShuffledTracesTB_dict = shuffledDiffMeans_byTB
sigDiff_means_TB = sigDiff_timebins
prob_aboveThresh_catch = prob_plus1mil_all_catches
prob_aboveThresh_miss = prob_plus1mil_all_misses
TGB_bucket = TGB_bucket_raw
baseline_len = baseline_buckets
plots_dir = plots_folder
todays_dt = todays_datetime
def plot_all_animals_pooled_BSF_TP(prey_type_str, threshold_str, catches_basesubfilt, misses_basesubfilt, catches_FBS_mean, misses_FBS_mean, edgeScoresTB_dict, ShuffledTracesTB_dict, sigDiff_means_TB, prob_aboveThresh_catch, prob_aboveThresh_miss, TGB_bucket, baseline_len, plots_dir, todays_dt): 
    img_type = ['.png', '.pdf']
    ### POOL ACROSS ANIMALS ### 
    allA_catches_mean_N = [] #(mean, N)
    allA_misses_mean_N = [] #(mean, N)
    for animal in catches_basesubfilt: 
        this_animal_N_TS = len(catches_basesubfilt[animal])
        if this_animal_N_TS != 0:
            this_animal_mean = catches_FBS_mean[animal]
            allA_catches_mean_N.append((this_animal_mean, this_animal_N_TS))
    for animal in misses_basesubfilt: 
        this_animal_N_TS = len(misses_basesubfilt[animal])
        if this_animal_N_TS != 0:
            this_animal_mean = misses_FBS_mean[animal]
            allA_misses_mean_N.append((this_animal_mean, this_animal_N_TS))
    # combined mean
    catches_combined_mean = list(itertools.starmap(mult_to_list, allA_catches_mean_N))
    catches_combined_mean_filtered = list(filter(lambda x: isinstance(x, list), catches_combined_mean))
    catches_combined_mean_num = [sum(x) for x in zip(*catches_combined_mean_filtered)]
    catches_combined_N = sum([x[1] for x in allA_catches_mean_N])
    catches_combined_mean = [x/catches_combined_N for x in catches_combined_mean_num]
    misses_combined_mean = list(itertools.starmap(mult_to_list, allA_misses_mean_N))
    misses_combined_mean_filtered = list(filter(lambda x: isinstance(x, list), misses_combined_mean))
    misses_combined_mean_num = [sum(x) for x in zip(*misses_combined_mean_filtered)]
    misses_combined_N = sum([x[1] for x in allA_misses_mean_N])
    misses_combined_mean = [x/misses_combined_N for x in misses_combined_mean_num]
    # combined percentage of edge count increase > threshold
    prob_aboveThresh_catch_allA = {}
    prob_aboveThresh_miss_allA = {}
    for timebin in prob_aboveThresh_catch:
        above_thresh_count_catch = []
        above_thresh_count_miss = []
        for animal in prob_aboveThresh_catch[timebin]:
            N_catches_above_thresh = prob_aboveThresh_catch[timebin][animal]*len(catches_basesubfilt[animal])
            above_thresh_count_catch.append(N_catches_above_thresh)
            N_misses_above_thresh = prob_aboveThresh_miss[timebin][animal]*len(misses_basesubfilt[animal])
            above_thresh_count_miss.append(N_misses_above_thresh)
        prob_aboveThresh_catch_allA[timebin] = sum(above_thresh_count_catch)/catches_combined_N
        prob_aboveThresh_miss_allA[timebin] = sum(above_thresh_count_miss)/misses_combined_N
    # calculate standard dev for combined means
    allA_Cdeviances = []
    allA_Mdeviances = []
    for animal in catches_basesubfilt:
        for trial in catches_basesubfilt[animal]:
            thisA_Cdeviance = (trial - catches_combined_mean)**2
            allA_Cdeviances.append(thisA_Cdeviance)
        for trial in misses_basesubfilt[animal]:
            thisA_Mdeviance = (trial - misses_combined_mean)**2
            allA_Mdeviances.append(thisA_Mdeviance)
    catches_std = np.sqrt(sum(allA_Cdeviances)/catches_combined_N)
    upper_bound_C = catches_combined_mean + catches_std
    lower_bound_C = catches_combined_mean - catches_std
    misses_std = np.sqrt(sum(allA_Mdeviances)/misses_combined_N)
    upper_bound_M = misses_combined_mean + misses_std
    lower_bound_M = misses_combined_mean - misses_std
    # differences of means bootstrapping
    ObservedDiff = np.array(catches_combined_mean) - np.array(misses_combined_mean)
    
    # correct for multiple/pointwise comparisons
    
    upper_bound_corrected = np.max(shuffMeans_traces, axis=0)
    lower_bound_corrected = np.min(ShuffledTraces_1000, axis=0)
    # set fig path and title
    figure_name = 'CannyEdgeDetector_BaselineSubtracted_SavGolFiltered_WithThreshProb_'+ prey_type_str + 'Trials_AllAnimals' + todays_dt + img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Mean change from baseline in number of edges in ROI on cuttlefish mantle during tentacle shots, as detected by Canny Edge Detector \n 95% Confidence intervals (calculated via shuffle tests, 20000 shuffles) plotted as shaded regions around each mean \n Baseline: mean of edge counts from t=0 to t=" + str(baseline_len/60) + " seconds \n Pooled across all animals, Prey movement type: " + prey_type_str + "\n Number of catches: " + str(catches_combined_N) + ", Number of misses: " + str(misses_combined_N)
    # draw fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    # subplot: real data and std 
    plt.subplot(2,1,1)
    plt.title('Observed data', fontsize=10, color='grey', style='italic')
    plt.ylabel("Change from baseline in number of edges")
    plot_xticks = np.arange(0, len(catches_combined_mean), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.ylim(-1500000,4000000)
    #plt.xlim(0,180)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    ymin, ymax = plt.ylim()
    # set colors
    color_meanM = [1.0, 0.0, 0.0, 0.8]
    color_stdM = [0.9, 0.0, 0.0, 0.1]
    color_meanC = [0.0, 0.0, 1.0, 0.8]
    color_stdC = [0.0, 0.0, 0.9, 0.1]
    color_pointwise95CI = [0.0, 0.5, 0.0, 1.0]
    color_global95CI = [1.0, 0.65, 0.0, 1.0]
    color_obsDiffMeans = [0.0, 0.0, 0.0, 1.0]
    # plot mean of catches and misses
    x_tbs = range(360)
    plt.plot(misses_combined_mean, linewidth=2, color=color_meanM, label='Miss')
    plt.fill_between(x_tbs, upper_bound_M, lower_bound_M, color=color_stdM)
    plt.plot(catches_combined_mean, linewidth=2, color=color_meanC, label='Catch')
    plt.fill_between(x_tbs, upper_bound_C, lower_bound_C, color=color_stdC)
    # label events
    for timebin in prob_aboveThresh_miss_allA:
        plt.plot(timebin, misses_combined_mean[timebin], 'ro')
        plt.plot((timebin, timebin), (ymin, ymax), 'k--', linewidth=0.5)
        plt.text(timebin, misses_combined_mean[timebin]-1000000, '%.1f'%(timebin/60 - 3)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_miss_allA[timebin]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
    for timebin in prob_aboveThresh_catch_allA:
        plt.plot(timebin, catches_combined_mean[timebin], 'bo')
        plt.text(timebin, catches_combined_mean[timebin]+1000000, '%.1f'%(timebin/60 - 3)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_catch_allA[timebin]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.35'))
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-600000, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-250000, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    #subplot: difference of observed means vs shuffled diff of means
    plt.subplot(2,1,2)
    plt.title('Significance of the Difference of means (catch vs miss)', fontsize=10, color='grey', style='italic')
    plt.ylabel("Difference of means in number of eges")
    plot_xticks = np.arange(0, len(catches_combined_mean), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.ylim(-1500000,4000000)
    #plt.xlim(0,180)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    ymin, ymax = plt.ylim()
    # plot differences of means, shuffled
    plt.plot(pointwise_upperbound, linestyle='--', color=color_pointwise95CI)
    plt.plot(pointwise_lowerbound, linestyle='--', color=color_pointwise95CI)
    # plot corrected bounds for p<0.05
    plt.plot(upper_bound_corrected, linestyle='--', color=color_global95CI)
    plt.plot(lower_bound_corrected, linestyle='--', color=color_global95CI)
    # plot real diff of means
    plt.plot(ObservedDiff, linewidth=2, linestyle='-', color=color_obsDiffMeans)

    for timebin in range(len(sigDiff_means_TB)):
        tb = sigDiff_means_TB[timebin]
        if tb < 180:
            plt.plot((tb, tb), (ymin, ymax), color='orange', linestyle='--', linewidth=1)
            plt.text(tb, catches_combined_mean[tb]+550000+(timebin*170000), "At {s:.1f} sec,\n p(diff of means)={p:.4f}".format(s=tb/60, p=pval_eachTB[tb]), fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.35'))
        else:
            plt.plot((tb, tb), (ymin, ymax), color='orange', linestyle='--', linewidth=1)
            plt.text(tb, catches_combined_mean[tb]+1750000, "At {s:.1f} sec, p(diff of means)={p:.4f}".format(s=tb/60, p=pval_eachTB[tb]), fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.35'))


    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()





def plot_pool_all_animals(prey_type, prey_type_str, catches_norm, misses_norm, catches_basesub_avg, misses_basesub_avg, catches_std, misses_std, TGB_bucket, plots_dir, todays_dt): 
    ### POOL ACROSS ANIMALS ### 
    all_catches = [] #(normed_avg, number_of_catches)
    all_misses = [] #(normed_avg, number_of_misses)
    catches_std_sq = {} 
    misses_std_sq = {} 
    for canny_type in range(len(prey_type)):
        for animal in prey_type[canny_type]: 
            this_animal_N_TS = len(prey_type[canny_type][animal])
            if this_animal_N_TS != 0:
                if canny_type == 0:
                    this_animal_M = catches_basesub_avg[animal]
                    this_animal_std = catches_std[animal][0]
                    all_catches.append((this_animal_M, this_animal_N_TS))
                    catches_std_sq[animal] = (this_animal_std**2, this_animal_N_TS)
                else: 
                    this_animal_M = misses_basesub_avg[animal]
                    this_animal_std = misses_std[animal][0]
                    all_misses.append((this_animal_M, this_animal_N_TS))
                    misses_std_sq[animal] = (this_animal_std**2, this_animal_N_TS)
    # combined mean
    catches_combined_mean = list(itertools.starmap(mult_to_list, all_catches))
    catches_combined_mean_filtered = list(filter(lambda x: isinstance(x, list), catches_combined_mean))
    catches_combined_mean_num = [sum(x) for x in zip(*catches_combined_mean_filtered)]
    catches_combined_N = sum([x[1] for x in all_catches])
    catches_combined_mean = [x/catches_combined_N for x in catches_combined_mean_num]
    misses_combined_mean = list(itertools.starmap(mult_to_list, all_misses))
    misses_combined_mean_filtered = list(filter(lambda x: isinstance(x, list), misses_combined_mean))
    misses_combined_mean_num = [sum(x) for x in zip(*misses_combined_mean_filtered)]
    misses_combined_N = sum([x[1] for x in all_misses])
    misses_combined_mean = [x/misses_combined_N for x in misses_combined_mean_num]
    # deviations per animal
    catches_deviations_sq = {}
    misses_deviations_sq = {}
    for canny_type in range(len(prey_type)):
        for animal in prey_type[canny_type]: 
            if len(prey_type[canny_type][animal]) != 0:
                if canny_type == 0:
                    this_deviation = catches_basesub_avg[animal] - catches_combined_mean
                    catches_deviations_sq[animal] = this_deviation**2
                else:
                    this_deviation = misses_basesub_avg[animal] - misses_combined_mean
                    misses_deviations_sq[animal] = this_deviation**2
    # combined std (formula from https://youtu.be/SHqWL6G3E08)
    # numerator: n*(std^2 + deviation^2) for each animal
    # denominator: N (total trials across all animals (sum of all n))
    # then sqrt the whole thing
    catches_numerator = {}
    misses_numerator = {}
    for canny_type in range(len(prey_type)):
        for animal in prey_type[canny_type]:
            if len(prey_type[canny_type][animal]) != 0:
                if canny_type == 0:
                    this_numerator = catches_std_sq[animal][1]*(catches_std_sq[animal][0] + catches_deviations_sq[animal])
                    catches_numerator[animal] = this_numerator
                else:
                    this_numerator = misses_std_sq[animal][1]*(misses_std_sq[animal][0] + misses_deviations_sq[animal])
                    misses_numerator[animal] = this_numerator
    all_catches_std_sq = np.nansum(np.array(list(catches_numerator.values())), axis=0)/catches_combined_N
    all_catches_std = [math.sqrt(x) for x in all_catches_std_sq]
    all_misses_std_sq = np.nansum(np.array(list(misses_numerator.values())), axis=0)/misses_combined_N
    all_misses_std = [math.sqrt(x) for x in all_misses_std_sq]
    z_val = 1.96 # Z value for 95% confidence interval
    error_all_catches = z_val*(all_catches_std/np.sqrt(catches_combined_N))
    error_all_misses = z_val*(all_misses_std/np.sqrt(misses_combined_N))

    figure_name = 'CannyEdgeDetector_'+ prey_type_str + 'Trials_AllAnimals_PercentChange_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Average change from baseline in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Baseline: mean of edge counts from t=0 to t=2 seconds \n Pooled across all animals, Prey movement type: " + prey_type_str + "\n Number of catches: " + str(catches_combined_N) + ", Number of misses: " + str(misses_combined_N)

    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.ylabel("Change from baseline in number of edges")
    plot_xticks = np.arange(0, len(catches_combined_mean), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.xlabel("Seconds")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')

    plt.plot(misses_combined_mean, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
    plt.fill_between(range(len(misses_combined_mean)), misses_combined_mean-error_all_misses, misses_combined_mean+error_all_misses, color=[1.0, 0.0, 0.0, 0.3])

    plt.plot(catches_combined_mean, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
    plt.fill_between(range(len(catches_combined_mean)), catches_combined_mean-error_all_catches, catches_combined_mean+error_all_catches, color=[0.0, 0.0, 1.0, 0.3])

    ymin, ymax = plt.ylim()
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket-5, ymax-5, "Tentacles Go Ballistic (TGB)", fontsize='x-small', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')

    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def normedRaw_Hist(dict_to_plot, data_type_str, plots_dir, todays_dt):
    for animal in dict_to_plot:
        figure_name = 'NormedRawHist_'+ animal + '_' + data_type_str + '_' + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = "Frequency histogram of baseline subtracted edge counts, raw \n Data type: " + data_type_str + "\n Baseline: mean of edge counts from t=0 to t=2 seconds \n Animal: " + animal
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        plt.ylabel("Frequency")
        plt.xlabel("Raw edge counts")
        plt.grid(b=True, which='major', linestyle='-')
        plt.hist(dict_to_plot[animal])
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def zScore_Hist(dict_to_plot, data_type_str, plots_dir, todays_dt):
    for animal in dict_to_plot:
        figure_name = 'zScoreHist_'+ animal + '_' + data_type_str + '_' + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = "Frequency histogram of baseline subtracted edge counts, z-scored \n Data type: " + data_type_str + "\n Baseline: mean of edge counts from t=0 to t=2 seconds \n Animal: " + animal
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        plt.ylabel("Frequency")
        plt.xlabel("z-score")
        plt.grid(b=True, which='major', linestyle='-')
        plt.hist(dict_to_plot[animal])
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
canny_counts_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\CannyCount_csv_smallCrop_Canny2000-7500"
plots_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\plots"

# in canny_counts_folder, list all csv files for TGB moments ("Tentacles Go Ballistic")
TGB_all = glob.glob(canny_counts_folder + os.sep + "*.csv")

# categorize tentacle shots according to prey movement
TGB_natural = []
TGB_patterned = []
TGB_causal = []
for TGB_file in TGB_all: 
    csv_name = TGB_file.split(os.sep)[-1]
    trial_date = csv_name.split('_')[2]
    trial_datetime = datetime.datetime.strptime(trial_date, '%Y-%m-%d')
    if trial_datetime < datetime.datetime(2014, 9, 13, 0, 0):
        TGB_natural.append(TGB_file)
    elif trial_datetime > datetime.datetime(2014, 10, 18, 0, 0):
        TGB_causal.append(TGB_file)
    else: 
        TGB_patterned.append(TGB_file)

# organize canny count data
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
# sav-gol filter and baseline subtract
savgol_window = 15
allTS_filtBaseSub = filtered_basesub_count(all_TS, 'all', baseline_buckets, savgol_window)
allCatches_filtBaseSub = filtered_basesub_count(all_catches, 'all', baseline_buckets, savgol_window)
allMisses_filtBaseSub = filtered_basesub_count(all_misses, 'all', baseline_buckets, savgol_window)
# zscore each animal so that I can pool all trials into a "superanimal"
allTS_filtBaseSub_Zscored = zScored_count(allTS_filtBaseSub, allTS_filtBaseSub)
allCatches_filtBaseSub_Zscored = zScored_count(allCatches_filtBaseSub, allTS_filtBaseSub)
allMisses_filtBaseSub_Zscored = zScored_count(allMisses_filtBaseSub, allTS_filtBaseSub)

########################################################
### ------------ STATS FOR SIGNIFICANCE ------------ ###
########################################################
#### shuffle test ####
No_of_Shuffles = 20000

### POOL ACROSS ALL ANIMALS, zscored, full trial
allA_ZfullTrial_C, allA_ZfullTrial_C_N, allA_ZfullTrial_M, allA_ZfullTrial_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 0, -1, "all")
# all animals full trial shuffle test
allA_ZfullTrial_pval, allA_ZfullTrial_025p, allA_ZfullTrial_975p, allA_ZfullTrial_mean = shuffle_test(allA_ZfullTrial_C, allA_ZfullTrial_M, No_of_Shuffles, "AllCatches-Zscored-fullTrial", "AllMisses-Zscored-fullTrial", allA_ZfullTrial_C_N, allA_ZfullTrial_M_N, True, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, before and after TGB
allA_ZpreTGB_C, allA_ZpreTGB_C_N, allA_ZpreTGB_M, allA_ZpreTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 0, TGB_bucket_raw-1, "all")
allA_ZpostTGB_C, allA_ZpostTGB_C_N, allA_ZpostTGB_M, allA_ZpostTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, TGB_bucket_raw, -1, "all")
# all animals preTGB shuffle test
allA_ZpreTGB_pval, allA_ZpreTGB_025p, allA_ZpreTGB_975p, allA_ZpreTGB_mean = shuffle_test(allA_ZpreTGB_C, allA_ZpreTGB_M, No_of_Shuffles, "AllCatch-Zscored-preTGB", "AllMiss-Zscored-preTGB", allA_ZpreTGB_C_N, allA_ZpreTGB_M_N, True, plots_folder, todays_datetime)
# all animals postTGB shuffle test
allA_ZpostTGB_pval, allA_ZpostTGB_025p, allA_ZpostTGB_975p, allA_ZpostTGB_mean = shuffle_test(allA_ZpostTGB_C, allA_ZpostTGB_M, No_of_Shuffles, "AllCatch-Zscored-postTGB", "AllMiss-Zscored-postTGB", allA_ZpostTGB_C_N, allA_ZpostTGB_M_N, True, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, baseline period
allA_Zbaseline_C, allA_Zbaseline_C_N, allA_Zbaseline_M, allA_Zbaseline_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 0, baseline_buckets, "all")
# all animals baseline shuffle test, number of tests = 2
allA_Zbaseline_pval, allA_Zbaseline_025p, allA_Zbaseline_975p, allA_Zbaseline_mean = shuffle_test(allA_Zbaseline_C, allA_Zbaseline_M, No_of_Shuffles, "AllCatch-Zscored-baseline", "AllMiss-Zscored-baseline", allA_Zbaseline_C_N, allA_Zbaseline_M_N, True, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, 0.5 seconds before TGB
allA_ZhalfSecPreTGB_C, allA_ZhalfSecPreTGB_C_N, allA_ZhalfSecPreTGB_M, allA_ZhalfSecPreTGB_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, baseline_buckets, TGB_bucket_raw-1, "all")
# all animals baseline shuffle test, number of tests = 2
allA_ZhalfSecPreTGB_pval, allA_ZhalfSecPreTGB_025p, allA_ZhalfSecPreTGB_975p, allA_ZhalfSecPreTGB_mean = shuffle_test(allA_ZhalfSecPreTGB_C, allA_ZhalfSecPreTGB_M, No_of_Shuffles, "AllCatch-Zscored-halfSecPreTGB", "AllMiss-Zscored-halfSecPreTGB", allA_ZhalfSecPreTGB_C_N, allA_ZhalfSecPreTGB_M_N, True, plots_folder, todays_datetime)
### individual animals, 0.5 seconds before TGB (show trends in individual animals to double check the pooled shuffle test)
indivA_N_shuffles = 1000
byA_ZhalfSecPreTGB_C, byA_ZhalfSecPreTGB_M = pool_timebins_byAnimal(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, baseline_buckets, TGB_bucket_raw-1)
byA_ZhalfSecPreTGB_pval = {}
byA_ZhalfSecPreTGB_025p = {}
byA_ZhalfSecPreTGB_975p = {}
byA_ZhalfSecPreTGB_mean = {}
for animal in byA_ZhalfSecPreTGB_C:
    thisA_ZhalfSecPreTGB_pval, thisA_ZhalfSecPreTGB_025p, thisA_ZhalfSecPreTGB_975p, thisA_ZhalfSecPreTGB_mean = shuffle_test(byA_ZhalfSecPreTGB_C[animal], byA_ZhalfSecPreTGB_M[animal], indivA_N_shuffles, animal+"-Zscored-halfSecPreTGB", animal+"-Zscored-halfSecPreTGB", len(allCatches_filtBaseSub_Zscored[animal]), len(allMisses_filtBaseSub_Zscored[animal]), True, plots_folder, todays_datetime)
    byA_ZhalfSecPreTGB_pval[animal] = thisA_ZhalfSecPreTGB_pval
    byA_ZhalfSecPreTGB_025p[animal] = thisA_ZhalfSecPreTGB_025p
    byA_ZhalfSecPreTGB_975p[animal] = thisA_ZhalfSecPreTGB_975p
    byA_ZhalfSecPreTGB_mean[animal] = thisA_ZhalfSecPreTGB_mean

### POOL ACROSS ALL ANIMALS, from 3.4 seconds after TGB to end
allA_Ztb205toEnd_C, allA_Ztb205toEnd_C_N, allA_Ztb205toEnd_M, allA_Ztb205toEnd_M_N = pool_acrossA_acrossTB(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 205, -1, "all")
# all animals baseline shuffle test, number of tests = 2
allA_Ztb205toEnd_pval, allA_Ztb205toEnd_025p, allA_Ztb205toEnd_975p, allA_Ztb205toEnd_mean = shuffle_test(allA_Ztb205toEnd_C, allA_Ztb205toEnd_M, No_of_Shuffles, "AllCatch-Zscored-tb205toEnd", "AllMiss-Zscored-tb205toEnd", allA_Ztb205toEnd_C_N, allA_Ztb205toEnd_M_N, True, plots_folder, todays_datetime)
### individual animals, from 3.4 seconds after TGB to end (show trends in individual animals to double check the pooled shuffle test)
byA_ZpostTB205_C, byA_ZpostTB205_M = pool_timebins_byAnimal(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 205, -1)
byA_ZpostTB205_pval = {}
byA_ZpostTB205_025p = {}
byA_ZpostTB205_975p = {}
byA_ZpostTB205_mean = {}
for animal in byA_ZpostTB205_C:
    thisA_ZpostTB205_pval, thisA_ZpostTB205_025p, thisA_ZpostTB205_975p, thisA_ZpostTB205_mean = shuffle_test(byA_ZpostTB205_C[animal], byA_ZpostTB205_M[animal], indivA_N_shuffles, animal+"-Catches-Zscored-tb205toEnd", animal+"-Misses-Zscored-tb205toEnd", len(allCatches_filtBaseSub_Zscored[animal]), len(allMisses_filtBaseSub_Zscored[animal]), True, plots_folder, todays_datetime)
    byA_ZpostTB205_pval[animal] = thisA_ZpostTB205_pval
    byA_ZpostTB205_025p[animal] = thisA_ZpostTB205_025p
    byA_ZpostTB205_975p[animal] = thisA_ZpostTB205_975p
    byA_ZpostTB205_mean[animal] = thisA_ZpostTB205_mean

### POOL ACROSS ALL ANIMALS, make a shuffle test of every time bin
allA_Z_byTB_C, allA_Z_byTB_C_N, allA_Z_byTB_M, allA_Z_byTB_M_N = pool_acrossA_keepTemporalStructure(allCatches_filtBaseSub_Zscored, allMisses_filtBaseSub_Zscored, 0, -1, "all")
ZedgeScores_byTB = {}
for timebin in range(360):
    # collect all edge scores for each time bin
    ZedgeScores_byTB[timebin] = {'catch':[], 'miss':[], 'pval': None, 'sigma': None, 'mean': None}
    for trial in allA_Z_byTB_C:
        ZedgeScores_byTB[timebin]['catch'].append(trial[timebin])
    for trial in allA_Z_byTB_M:
        ZedgeScores_byTB[timebin]['miss'].append(trial[timebin])
    # shuffle test each time bin
    ZedgeScores_byTB[timebin]['pval'], ZedgeScores_byTB[timebin]['025p'], ZedgeScores_byTB[timebin]['975p'], ZedgeScores_byTB[timebin]['mean'] = shuffle_test(ZedgeScores_byTB[timebin]['catch'], ZedgeScores_byTB[timebin]['miss'], No_of_Shuffles, 'AllCatches-Zscored-TB'+str(timebin), 'AllMisses-Zscored-TB'+str(timebin), allA_Z_byTB_C_N, allA_Z_byTB_M_N, True, plots_folder, todays_datetime)

# pointwise p<0.05 bounds
pointwise005sig_upperbound = []
pointwise005sig_lowerbound = []
for timebin in sorted(ZedgeScores_byTB.keys()):
    thisTB_025p = ZedgeScores_byTB[timebin]['025p']
    thisTB_975p = ZedgeScores_byTB[timebin]['975p']
    pointwise005sig_upperbound.append(thisTB_975p)
    pointwise005sig_lowerbound.append(thisTB_025p)

# calculate real difference of mean catch and mean miss
allA_allC_Z = []
allA_allM_Z = []
for animal in allCatches_filtBaseSub_Zscored:
    for trial in allCatches_filtBaseSub_Zscored[animal]:
        allA_allC_Z.append(trial)
        allA_allM_Z.append(trial)
allA_allC_Z_mean = np.mean(allA_allC_Z, axis=0)
allA_allM_Z_mean = np.mean(allA_allM_Z, axis=0)
Observed_DiffMeans = allA_allC_Z_mean + allA_allM_Z_mean

# visualize
plt.plot(pointwise005sig_upperbound, 'g--')
plt.plot(pointwise005sig_lowerbound, 'g--')
plt.plot(Observed_DiffMeans, 'k-')
plt.show()

# generate random traces to correct threshold for p<0.05
shuffledDiffMeans_byTB = {}
for timebin in ZedgeScores_byTB:
    shuffledDiffMeans_byTB[timebin] = gen_shuffled_traces(ZedgeScores_byTB[timebin]['catch'], ZedgeScores_byTB[timebin]['miss'], 1000, len(ZedgeScores_byTB[timebin]['catch']), len(ZedgeScores_byTB[timebin]['miss']))
shuffMeans_traces = []
shuffMeans_traces_N = len(shuffledDiffMeans_byTB[0])
for st in range(shuffMeans_traces_N):
    this_trace = []
    for timebin in shuffledDiffMeans_byTB:
        this_trace.append(shuffledDiffMeans_byTB[timebin][st][0])
    shuffMeans_traces.append(this_trace)
shuffMeans_traces = np.array(shuffMeans_traces)
# check how many of these random traces violate the p<0.05 generated by timebin-wise shuffle test
outOfBounds_upper = [shuffMeans_traces[x]>pointwise005sig_upperbound[x] for trial in shuffMeans_traces for x in range(len(trial))]
outOfBounds_lower = [shuffMeans_traces[x]<pointwise005sig_lowerbound[x] for trial in shuffMeans_traces for x in range(len(trial))]
outOfBounds_upper_total = [sum(trial) for trial in outOfBounds_upper]
outOfBounds_lower_total = [sum(trial) for trial in outOfBounds_lower]

#######################################################
### ---------------- PLOT THE DATA ---------------- ###
#######################################################

## individual animals
plot_indiv_animals_BSF_TP("all", "1 million", all_catches_basesub_filtered, all_misses_basesub_filtered, all_catches_basesub_filtered_mean, all_misses_basesub_filtered_mean, prob_plus1mil_all_catches, prob_plus1mil_all_misses, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS
plot_all_animals_pooled_BSF_TP("all", "1 million", all_catches_basesub_filtered, all_misses_basesub_filtered, all_catches_basesub_filtered_mean, all_misses_basesub_filtered_mean, ZedgeScores_byTB, shuffledDiffMeans_byTB, prob_plus1mil_all_catches, prob_plus1mil_all_misses, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

########################################################
### ------ under construction!!!! ------ ###
########################################################

# calculate probability of basesubfilt'd edge counts increasing past a threshold
timebins_to_check = [205, 240, 300, 359]
threshold_catchVmiss = 1000000
prob_plus1mil_all_catches, prob_plus1mil_all_misses = prob_of_tb_above_edgecount_thresh(timebins_to_check, threshold_catchVmiss, all_catches_basesub_filtered, all_misses_basesub_filtered)


# pool all basesubfilt'd data to find distribution of data
allTS_basesub_filtered = []

for animal in all_catches_basesub_filtered:
    for trial in all_catches_basesub_filtered[animal]:
        for timebin in trial:
            allTS_basesub_filtered.append(timebin)
for animal in all_misses_basesub_filtered:
    for trial in all_misses_basesub_filtered[animal]:
        for timebin in trial:
            allTS_basesub_filtered.append(timebin)
# Z score full dataset, find mean and std
allTS_basesub_filtered_zScore = stats.zscore(allTS_basesub_filtered, ddof=1)
allTS_normed_zScore_mean = np.nanmean(allTS_normed_zScore)
allTS_normed_zScore_std = np.nanstd(allTS_normed_zScore, ddof=1)


# MIN MAX NORMALIZATION
# make min-max normalization of all tentacle shots for each animal
all_MinMaxNormed = {}
all_MinMaxNormed_preTGB = {}
all_MinMaxNormed_postTGB = {}
for animal in all_TS:
    min_thisAnimal = min(np.hstack(all_TS[animal]))
    max_thisAnimal = max(np.hstack(all_TS[animal]))
    all_MinMaxNormed[animal] = {'catches': [], 'misses': []}
    all_MinMaxNormed_preTGB[animal] = {'catches': [], 'misses': []}
    all_MinMaxNormed_postTGB[animal] = {'catches': [], 'misses': []}
    for trial in all_catches[animal]:
        rescaled_trial = [(x-min_thisAnimal)/(max_thisAnimal - min_thisAnimal) for x in trial]
        all_MinMaxNormed[animal]['catches'].append(rescaled_trial)
        all_MinMaxNormed_preTGB[animal]['catches'].append(rescaled_trial[:TGB_bucket_raw-1])
        all_MinMaxNormed_postTGB[animal]['catches'].append(rescaled_trial[TGB_bucket_raw:])
    for trial in all_misses[animal]:
        rescaled_trial = [(x-min_thisAnimal)/(max_thisAnimal - min_thisAnimal) for x in trial]
        all_MinMaxNormed[animal]['misses'].append(rescaled_trial)
        all_MinMaxNormed_preTGB[animal]['misses'].append(rescaled_trial[:TGB_bucket_raw-1])
        all_MinMaxNormed_postTGB[animal]['misses'].append(rescaled_trial[TGB_bucket_raw:])




diffs_of_baselines = {}
for animal in all_catches_baseline:
    diffs_of_baselines[animal] = all_catches_baseline[animal] - all_misses_baseline[animal]




basesub_filtered_count(nat_raw, "natural", baseline_buckets, savgol_window, nat_catches_baseline, nat_misses_baseline, nat_catches_basesub_filtered, nat_misses_basesub_filtered, nat_catches)
basesub_filtered_count(pat_raw, "patterned", baseline_buckets, pat_catches_baseline, pat_misses_baseline, pat_catches_basesub_filtered, pat_misses_basesub_filtered)
basesub_filtered_count(caus_raw, "causal", baseline_buckets, caus_catches_baseline, caus_misses_baseline, caus_catches_basesub_filtered, caus_misses_basesub_filtered)

all_basesub = [all_catches_basesub, all_misses_basesub_filtered]


nat_basesub_filtered = [nat_catches_basesub_filtered, nat_misses_basesub_filtered]
pat_basesub_filtered = [pat_catches_basesub_filtered, pat_misses_basesub_filtered]
caus_basesub_filtered = [caus_catches_basesub_filtered, caus_misses_basesub_filtered]







for animal in all_MinMaxNormed:
    figure_name = 'MinMaxNormed_'+ animal + "_" + todays_datetime + '.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = "Min-max normalized change in number of edges in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Animal: " + animal + "\n Number of catches: " + str(len(all_MinMaxNormed[animal]['catches'])) + ", Number of misses: " + str(len(all_MinMaxNormed[animal]['misses']))
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.ylabel("Min-max normalized change in number of edges")
    plot_xticks = np.arange(0, len(all_TS[animal][0]), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.xlabel("Seconds")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')
    for trial in all_MinMaxNormed[animal]['catches']:
        plt.plot(trial, color='blue', alpha=0.2, label='Catch')
    for trial in all_MinMaxNormed[animal]['misses']:
        plt.plot(trial, color='red', alpha=0.2, label='Miss')
    ymin, ymax = plt.ylim()
    plt.plot((TGB_bucket_raw, TGB_bucket_raw), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket_raw-5, ymax-5, "Tentacles Go Ballistic (TGB)", fontsize='x-small', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

allO_allA_postTGB = []
for animal in all_MinMaxNormed_postTGB:
    allObservations_thisAnimal = []
    for trial in all_MinMaxNormed_postTGB[animal]['catches']:
        for timebin in trial:
            allObservations_thisAnimal.append(timebin)
    for trial in all_MinMaxNormed_postTGB[animal]['misses']:
        for timebin in trial:
            allObservations_thisAnimal.append(timebin)
    allO_allA_postTGB.append(allObservations_thisAnimal)

# plot distribution of edge counts post TGB
allO_thisA_postTGB_mean = np.nanmean(np.hstack(allO_allA_postTGB))
allO_thisA_postTGB_std = np.nanstd(np.hstack(allO_allA_postTGB), ddof=1)
figure_name = 'MinMaxNormedHist_postTGB_allObservations_' + todays_datetime + '.png'
figure_path = os.path.join(plots_folder, figure_name)
figure_title = "Frequency histogram of Min-max normalized edge counts in cuttlefish mantle pattern AFTER tentacle shots, as detected by Canny Edge Detector \n All animals \n Number of catches: " + str(len(all_MinMaxNormed[animal]['catches'])) + ", Number of misses: " + str(len(all_MinMaxNormed[animal]['misses'])) + "\n Mean: " + str(allO_thisA_mean) + ", Standard dev: " + str(allO_thisA_std)
plt.figure(figsize=(16,9), dpi=200)
plt.suptitle(figure_title, fontsize=12, y=0.98)
plt.ylabel("Frequency")
plt.xlabel("Min-max normalized edge count")
#plt.xlabel("Frame number, original framerate = 60fps")
plt.grid(b=True, which='major', linestyle='-')
plt.hist(np.hstack(allO_allA_postTGB))
plt.savefig(figure_path)
plt.show(block=False)
plt.pause(1)
plt.close()

#f = Fitter(np.hstack(allO_allA_postTGB))
#f.fit()
#plt.hist(np.hstack(allO_allA_postTGB))
#f.summary()
#>>> f.summary()
##            sumsquare_error
##levy             129.317025
##invgauss         239.201950
##invgamma         248.078092
##halfcauchy       344.966041
##expon            360.504135
#plt.show()



# natural
plot_indiv_animals("natural", nat_catches, nat_catches_norm, nat_misses_norm, nat_catches_basesub_avg, nat_misses_basesub_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)
# patterned
plot_indiv_animals("patterned", pat_catches, pat_catches_norm, pat_misses_norm, pat_catches_basesub_avg, pat_misses_basesub_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)
# causal
plot_indiv_animals("causal", caus_catches, caus_catches_norm, caus_misses_norm, caus_catches_basesub_avg, caus_misses_basesub_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS ### 
# all
plot_pool_all_animals(all_raw, "all", all_catches_basesub, all_misses_basesub, all_catches_basesub_avg, all_misses_basesub_avg, all_catches_std_error, all_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# natural
plot_pool_all_animals(nat_raw "natural", nat_catches_norm, nat_misses_norm, nat_catches_basesub_avg, nat_misses_basesub_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# patterned
plot_pool_all_animals(pat_raw "patterned", pat_catches_norm, pat_misses_norm, pat_catches_basesub_avg, pat_misses_basesub_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# causal
plot_pool_all_animals(caus_raw "causal", caus_catches_norm, caus_misses_norm, caus_catches_basesub_avg, caus_misses_basesub_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)


# plot all traces
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        for trial in all_norm[canny_type][animal]:
            if canny_type == 0:
                plt.plot(trial, color='blue', alpha=0.1)
            else:
                plt.plot(trial, color='red', alpha=0.1)
plt.show()



# z-score for each animal, across all timebins and trials
allTS_normed_perAnimal = {}
allTS_normed_perAnimal_trials = {}
zScore_perAnimal = {}
zScore_perAnimal_trials = {}
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        for trial in all_norm[canny_type][animal]:
            for timebin in trial:
                allTS_normed_perAnimal.setdefault(animal,[]).append(timebin)
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        if canny_type == 0:
            allTS_normed_perAnimal_trials[animal] = [[],[]]
        for trial in all_norm[canny_type][animal]:
            allTS_normed_perAnimal_trials[animal][canny_type].append(trial)
for animal in allTS_normed_perAnimal:        
    allTS_normed_perAnimal[animal] = np.array(allTS_normed_perAnimal[animal])
    allTS_normed_perAnimal_trials[animal] = np.array(allTS_normed_perAnimal_trials[animal])
for animal in allTS_normed_perAnimal:
    zScore_perAnimal[animal] = stats.zscore(allTS_normed_perAnimal[animal], ddof=1)
    len_catches = len(allTS_normed_perAnimal_trials[animal][0])
    len_misses = len(allTS_normed_perAnimal_trials[animal][1])
    allTrials_thisAnimal = np.concatenate((allTS_normed_perAnimal_trials[animal][0], allTS_normed_perAnimal_trials[animal][1]), axis=0)
    zScore_thisAnimal = stats.zscore(allTrials_thisAnimal, axis=None, ddof=1)
    zScore_perAnimal_trials[animal] = np.array([zScore_thisAnimal[0:len_catches-1],zScore_thisAnimal[len_catches:]])
## visualize
normedRaw_Hist(allTS_normed_perAnimal, "AcrossAllTimebinsAndTrials", plots_folder, todays_datetime)
normedRaw_Hist(allTS_normed_perAnimal_trials, "AcrossAllTimebinsByTrial", plots_folder, todays_datetime)
zScore_Hist(zScore_perAnimal, "AcrossAllTimebinsAndTrials", plots_folder, todays_datetime)
zScore_Hist(zScore_perAnimal_trials, "AcrossAllTimebinsByTrial", plots_folder, todays_datetime)
## find mean zscore of each trial
for animal in zScore_perAnimal_trials: 
    for canny_type in zScore_perAnimal_trials[animal]:
        for 
        plt.plot(trial)
    plt.show()

# zscore within each animal, for each time bin, pooled across all trials
allTS_normed_perA_perTB = {}
zScore_perA_perTB = {}
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        for trial in all_norm[canny_type][animal]:
            for timebin in range(len(trial)):
                allTS_normed_perA_perTB.setdefault(animal,{}).setdefault(timebin,[]).append(trial[timebin])
for animal in allTS_normed_perA_perTB:
    for timebin in allTS_normed_perA_perTB[animal]:
        allTS_normed_perA_perTB[animal][timebin] = np.array(allTS_normed_perA_perTB[animal][timebin])
for animal in allTS_normed_perA_perTB:
    zScore_perA_perTB[animal] = {}
    for timebin in allTS_normed_perA_perTB[animal]:
        zScore_perA_perTB[animal][timebin] = stats.zscore(allTS_normed_perA_perTB[animal][timebin], axis=0, ddof=1)
## visualize








# plot individual animals
# downsample 
TGB_window = 5 
downsample_buckets = int(np.ceil(len(TGB_moment)/TGB_window))
TGB_smooth = []
counter = 0 
TGB_buckets = int(np.ceil((len(TGB_moment)/2)/TGB_window))
for bucket in range(downsample_buckets): 
    start = counter
    end = counter + TGB_window - 1
    this_bucket = np.mean(TGB_moment[start:end])
    TGB_smooth.append(this_bucket)
    counter = counter + TGB_window

## FIN