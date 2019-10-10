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

### FUNCTIONS ###
def categorize_by_animal(TGB_files, all_animals_dict):
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        all_animals_dict[TGB_animal].append(TGB_moment)

def categorize_by_animal_catchVmiss(TGB_files, catch_dict, miss_dict):
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        if TGB_type == "catch":
            catch_dict[TGB_animal].append(TGB_moment)
        if TGB_type == "miss": 
            miss_dict[TGB_animal].append(TGB_moment)

def basesub_filtered_count(prey_type, prey_type_str, baseline_len, savgol_filter_window, baseline_catch, baseline_miss, basesub_catch, basesub_miss, basesub_mean_catch, basesub_mean_miss):
    # make baseline for each animal, catch vs miss
    for canny_type in range(len(prey_type)): 
        for animal in prey_type[canny_type]: 
            try:
                # baseline subtract each trial, then apply sav-gol filter
                all_basesub_filtered_trials = []
                for trial in prey_type[canny_type][animal]:
                    TGB_baseline = np.nanmean(trial[0:baseline_len])
                    basesub_trial = [float(x-TGB_baseline) for x in trial]
                    basesub_trial_filtered = scipy.signal.savgol_filter(basesub_trial, savgol_filter_window, 3)
                    all_basesub_filtered_trials.append(basesub_trial_filtered)
                basesub_filtered_mean = np.nanmean(all_basesub_filtered_trials, axis=0)
                if canny_type == 0:
                    baseline_catch[animal] = TGB_baseline
                    basesub_catch[animal] = all_basesub_filtered_trials
                    basesub_mean_catch[animal] = basesub_filtered_mean
                if canny_type == 1:
                    baseline_miss[animal] = TGB_baseline
                    basesub_miss[animal] = all_basesub_filtered_trials
                    basesub_mean_miss[animal] = basesub_filtered_mean
            except Exception:
                if canny_type == 0:
                    print("{a} made no catches during {p} prey movement".format(a=animal,p=prey_type_str))
                if canny_type == 1:
                    print("{a} made no misses during {p} prey movement".format(a=animal,p=prey_type_str))

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

def plot_indiv_animals_BSF_TP(prey_type_str, threshold_str, catches_basesubfilt, misses_basesubfilt, catches_basesubfilt_mean, misses_basesubfilt_mean, prob_aboveThresh_catch, prob_aboveThresh_miss, TGB_bucket, baseline_len, plots_dir, todays_dt):
    # plot individual animals
    img_type = ['.png', '.pdf']
    for animal in catches_basesubfilt.keys(): 
        try:
            #canny_std_catch = np.nanstd(catches_basesubfilt[animal], axis=0, ddof=1)
            canny_N_catch = len(catches_basesubfilt[animal])
            #canny_std_miss = np.nanstd(misses_basesubfilt[animal], axis=0, ddof=1)
            canny_N_miss = len(misses_basesubfilt[animal])
            catches_mean = catches_basesubfilt_mean[animal]
            misses_mean = misses_basesubfilt_mean[animal]

            figure_name = 'CannyEdgeDetector_BaselineSubtracted_SavGolFiltered_WithThreshProb_'+ prey_type_str + 'Trials_' + animal + "_" + todays_dt + img_type[0]
            figure_path = os.path.join(plots_dir, figure_name)
            figure_title = "Mean change from baseline in number of edges in ROI on cuttlefish mantle during tentacle shots, as detected by Canny Edge Detector \n Individual trials plotted with more transparent traces \n Baseline: mean of edge counts from t=0 to t=" + str(baseline_len/60) + " seconds \n Prey Movement type: " + prey_type_str + ", Animal: " + animal + "\n Number of catches: " + str(canny_N_catch) + ", Number of misses: " + str(canny_N_miss)
            plt.figure(figsize=(16,9), dpi=200)
            plt.suptitle(figure_title, fontsize=12, y=0.99)
            plt.ylabel("Change from baseline in number of edges")
            plot_xticks = np.arange(0, len(catches_basesubfilt_mean[animal]), step=60)
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
                plt.text(timebin, misses_mean[timebin]-1000000, '%.1f'%(timebin/60)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_miss[timebin][animal]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
            for timebin in prob_aboveThresh_catch:
                plt.plot(timebin, catches_mean[timebin], 'bo')
                plt.text(timebin, catches_mean[timebin]+1000000, '%.1f'%(timebin/60)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_catch[timebin][animal]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.35'))

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
                for binned_count in trial[timebin_start:timebin_end]:
                    pooled_catches.append(binned_count)
        else:
            print('{a} made no catch tentacle shots during {p} prey movement'.format(a=animal, p=prey_type_str))
        thisA_missesN = len(misses_dict[animal])
        pooled_misses_Ntrials = pooled_misses_Ntrials + thisA_missesN
        if thisA_missesN != 0:
            for trial in misses_dict[animal]:
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
            for binned_count in trial[start_tb:end_tb]:
                pooledTB_byA_catches[animal].append(binned_count)
        for trial in misses_dict[animal]:
            for binned_count in trial[start_tb:end_tb]:
                pooledTB_byA_misses[animal].append(binned_count)
    return pooledTB_byA_catches, pooledTB_byA_misses

def shuffle_test(Group1, Group2, N_Shuffles, Group1_str, Group2_str, Group1_N, Group2_N, plots_dir, todays_dt):
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
    sigma_shuff = (OPerf - np.mean(SPerf))/np.std(SPerf, ddof=1)
    # show histogram of diffs of shuffled means
    figure_name = 'ShuffleTest_'+ Group1_str + '_' + Group2_str + '_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Histogram of the differences in means of randomly labeled data, Number of shuffles = " + str(N_Shuffles) + "\n Group 1: " + Group1_str + ", N = " + str(Group1_N) + "\n Group 2: " + Group2_str + ", N = " + str(Group2_N) + "\n P-value of shuffle test: " + str(pVal) + ", Sigma of shuffle test: " + str(sigma_shuff)
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.hist(SPerf)
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    plt.plot((OPerf, OPerf), (ymin, ymax), 'g--', linewidth=1)
    plt.text(OPerf, ymax-5, "Difference of Labeled Means = " + str(OPerf), fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return pVal, sigma_shuff

def mult_to_list(input_list, multiplier):
    if not np.isnan(input_list).any(): 
        return [x*multiplier for x in input_list]

def plot_all_animals_pooled_BSF_TP(prey_type_str, threshold_str, catches_basesubfilt, misses_basesubfilt, catches_basesubfilt_mean, misses_basesubfilt_mean, prob_aboveThresh_catch, prob_aboveThresh_miss, TGB_bucket, baseline_len, plots_dir, todays_dt): 
    img_type = ['.png', '.pdf']
    ### POOL ACROSS ANIMALS ### 
    allA_catches_mean_N = [] #(mean, N)
    allA_misses_mean_N = [] #(mean, N)
    for animal in catches_basesubfilt: 
        this_animal_N_TS = len(catches_basesubfilt[animal])
        if this_animal_N_TS != 0:
            this_animal_mean = catches_basesubfilt_mean[animal]
            allA_catches_mean_N.append((this_animal_mean, this_animal_N_TS))
    for animal in misses_basesubfilt: 
        this_animal_N_TS = len(misses_basesubfilt[animal])
        if this_animal_N_TS != 0:
            this_animal_mean = misses_basesubfilt_mean[animal]
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

    figure_name = 'CannyEdgeDetector_BaselineSubtracted_SavGolFiltered_WithThreshProb_'+ prey_type_str + 'Trials_AllAnimals' + todays_dt + img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Mean change from baseline in number of edges in ROI on cuttlefish mantle during tentacle shots, as detected by Canny Edge Detector \n Individual trials plotted with more transparent traces \n Baseline: mean of edge counts from t=0 to t=" + str(baseline_len/60) + " seconds \n Pooled across all animals, Prey movement type: " + prey_type_str + "\n Number of catches: " + str(catches_combined_N) + ", Number of misses: " + str(misses_combined_N)

    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.ylabel("Change from baseline in number of edges")
    plot_xticks = np.arange(0, len(catches_combined_mean), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.ylim(-1500000,4000000)
    #plt.xlim(0,180)
    plt.xlabel("Seconds")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')
    ymin, ymax = plt.ylim()

    for animal in misses_basesubfilt:
        for trial in misses_basesubfilt[animal]:
            plt.plot(trial, linewidth=1, color=[1.0, 0.0, 0.0, 0.1])
        for trial in catches_basesubfilt[animal]:
            plt.plot(trial, linewidth=1, color=[0.0, 0.0, 1.0, 0.1])
    plt.plot(misses_combined_mean, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
    plt.plot(catches_combined_mean, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
    for timebin in prob_aboveThresh_miss_allA:
        plt.plot(timebin, misses_combined_mean[timebin], 'ro')
        plt.plot((timebin, timebin), (ymin, ymax), 'k--', linewidth=1)
        plt.text(timebin, misses_combined_mean[timebin]-1000000, '%.1f'%(timebin/60)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_miss_allA[timebin]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
    for timebin in prob_aboveThresh_catch_allA:
        plt.plot(timebin, catches_combined_mean[timebin], 'bo')
        plt.text(timebin, catches_combined_mean[timebin]+1000000, '%.1f'%(timebin/60)+" seconds after TGB, \n"+'%.2d'%(prob_aboveThresh_catch_allA[timebin]*100)+"% trials increased \nby > "+threshold_str+" edges", fontsize='x-small', ha='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.35'))

    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-600000, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-250000, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')

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
# by animal
all_TS = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
# all, by catches v misses
all_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_catches_baseline = {}
all_misses_baseline = {}
all_catches_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_misses_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_catches_basesub_filtered_mean = {}
all_misses_basesub_filtered_mean = {}
# natural, by catches v misses
nat_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_baseline = {}
nat_misses_baseline = {}
nat_catches_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_basesub_filtered_mean = {}
nat_misses_basesub_filtered_mean = {}
# patterned, by catches v misses
pat_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_baseline = {}
pat_misses_baseline = {}
pat_catches_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_basesub_filtered_mean = {}
pat_misses_basesub_filtered_mean = {}
# causal, by catches v misses
caus_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_baseline = {}
caus_misses_baseline = {}
caus_catches_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses_basesub_filtered = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_basesub_filtered_mean = {}
caus_misses_basesub_filtered_mean = {}

# collect all canny counts and categorize by animal
categorize_by_animal(TGB_all, all_TS)
# collect all canny counts and categorize by animal and type (catch vs miss)
categorize_by_animal_catchVmiss(TGB_all, all_catches, all_misses)
categorize_by_animal_catchVmiss(TGB_natural, nat_catches, nat_misses)
categorize_by_animal_catchVmiss(TGB_patterned, pat_catches, pat_misses)
categorize_by_animal_catchVmiss(TGB_causal, caus_catches, caus_misses)
# organize by prey type
all_raw = [all_catches, all_misses]
nat_raw= [nat_catches, nat_misses]
pat_raw= [pat_catches, pat_misses]
caus_raw= [caus_catches, caus_misses]
# time bin for moment tentacles go ballistic
TGB_bucket_raw = 180

########################################################
### ------ DATA NORMALIZATION/STANDARDIZATION ------ ###
########################################################

# BASELINE SUBTRACTION 
baseline_buckets = 150
# baseline subtract and sav-gol filter
savgol_window = 15
#basesub_filtered_count(prey_type, prey_type_str, baseline_len, savgol_filter_window, baseline_catch, baseline_miss, basesub_catch, basesub_miss)
basesub_filtered_count(all_raw, "all", baseline_buckets, savgol_window, all_catches_baseline, all_misses_baseline, all_catches_basesub_filtered, all_misses_basesub_filtered, all_catches_basesub_filtered_mean, all_misses_basesub_filtered_mean)
# calculate probability of basesubfilt'd edge counts increasing by at least 1 mil
timebins_to_check = [205, 240, 300, 359]
threshold_catchVmiss = 1000000
prob_plus1mil_all_catches, prob_plus1mil_all_misses = prob_of_tb_above_edgecount_thresh(timebins_to_check, threshold_catchVmiss, all_catches_basesub_filtered, all_misses_basesub_filtered)
## visualize the data
plot_indiv_animals_BSF_TP("all", "1 million", all_catches_basesub_filtered, all_misses_basesub_filtered, all_catches_basesub_filtered_mean, all_misses_basesub_filtered_mean, prob_plus1mil_all_catches, prob_plus1mil_all_misses, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS
plot_all_animals_pooled_BSF_TP("all", "1 million", all_catches_basesub_filtered, all_misses_basesub_filtered, all_catches_basesub_filtered_mean, all_misses_basesub_filtered_mean, prob_plus1mil_all_catches, prob_plus1mil_all_misses, TGB_bucket_raw, baseline_buckets, plots_folder, todays_datetime)

########################################################
### ------------ STATS FOR SIGNIFICANCE ------------ ###
########################################################

# shuffle test
allTS_basesub_filtered = [all_catches_basesub_filtered, all_misses_basesub_filtered]
No_of_Shuffles = 20000

### POOL ACROSS ALL ANIMALS, full trial
allA_basesub_filt_catches, allA_basesub_filt_catches_N, allA_basesub_filt_misses, allA_basesub_filt_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, 0, -1, "all")
# all animals full trial shuffle test, number of tests = 16
pVal_allA_fullTrial_basesubfilt, sigma_allA_fullTrial_basesubfilt = shuffle_test(allA_basesub_filt_catches, allA_basesub_filt_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-fullTrial", "AllMiss-BaseSubFilt-fullTrial", allA_basesub_filt_catches_N, allA_basesub_filt_misses_N, plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, before and after TGB
allA_basesub_filt_preTGB_catches, allA_basesub_filt_preTGB_catches_N, allA_basesub_filt_preTGB_misses, allA_basesub_filt_preTGB_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, 0, TGB_bucket_raw-1, "all")
allA_basesub_filt_postTGB_catches, allA_basesub_filt_postTGB_catches_N, allA_basesub_filt_postTGB_misses, allA_basesub_filt_postTGB_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, TGB_bucket_raw, -1, "all")
# all animals preTGB shuffle test, number of tests = 9
pVal_allA_preTGB_basesubfilt, sigma_allA_preTGB_basesubfilt = shuffle_test(allA_basesub_filt_preTGB_catches, allA_basesub_filt_preTGB_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-preTGB", "AllMiss-BaseSubFilt-preTGB", allA_basesub_filt_preTGB_catches_N, allA_basesub_filt_preTGB_misses_N, plots_folder, todays_datetime)
# all animals postTGB shuffle test, number of tests = 6
pVal_allA_postTGB_basesubfilt, sigma_allA_postTGB_basesubfilt = shuffle_test(allA_basesub_filt_postTGB_catches, allA_basesub_filt_postTGB_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-postTGB", "AllMiss-BaseSubFilt-postTGB", allA_basesub_filt_postTGB_catches_N, allA_basesub_filt_postTGB_misses_N,  plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, baseline period
allA_basesub_filt_baseline_catches, allA_basesub_filt_baseline_catches_N, allA_basesub_filt_baseline_misses, allA_basesub_filt_baseline_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, 0, baseline_buckets, "all")
# all animals baseline shuffle test, number of tests = 2
pVal_allA_baseline_basesubfilt, sigma_allA_baseline_basesubfilt = shuffle_test(allA_basesub_filt_baseline_catches, allA_basesub_filt_baseline_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-baseline", "AllMiss-BaseSubFilt-baseline", allA_basesub_filt_baseline_catches_N, allA_basesub_filt_baseline_misses_N,  plots_folder, todays_datetime)

### POOL ACROSS ALL ANIMALS, 0.5 seconds before TGB
allA_basesub_filt_halfSecPreTGB_catches, allA_basesub_filt_halfSecPreTGB_catches_N, allA_basesub_filt_halfSecPreTGB_misses, allA_basesub_filt_halfSecPreTGB_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, baseline_buckets, TGB_bucket_raw-1, "all")
# all animals baseline shuffle test, number of tests = 2
pVal_allA_halfSecPreTGB_basesubfilt, sigma_allA_halfSecPreTGB_basesubfilt = shuffle_test(allA_basesub_filt_halfSecPreTGB_catches, allA_basesub_filt_halfSecPreTGB_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-halfSecPreTGB", "AllMiss-BaseSubFilt-halfSecPreTGB", allA_basesub_filt_halfSecPreTGB_catches_N, allA_basesub_filt_halfSecPreTGB_misses_N,  plots_folder, todays_datetime)
### individual animals, 0.5 seconds before TGB (to double check the pooled shuffle test)
indivA_N_shuffles = 1000
halfSecPreTGB_catches_basesubfilt_byAnimal, halfSecPreTGB_misses_basesubfilt_byAnimal = pool_timebins_byAnimal(all_catches_basesub_filtered, all_misses_basesub_filtered, baseline_buckets, TGB_bucket_raw-1)
pVal_halfSecPreTGB_basesubfilt_byAnimal = {}
sigma_halfSecPreTGB_basesubfilt_byAnimal = {}
for animal in halfSecPreTGB_catches_basesubfilt_byAnimal:
    pVal_thisA_halfSecPreTGB_basesubfilt, sigma_thisA_halfSecPreTGB_basesubfilt = shuffle_test(halfSecPreTGB_catches_basesubfilt_byAnimal[animal], halfSecPreTGB_misses_basesubfilt_byAnimal[animal], indivA_N_shuffles, animal+"-BaseSubFilt-halfSecPreTGB", animal+"-BaseSubFilt-halfSecPreTGB", len(all_catches_basesub_filtered[animal]), len(all_misses_basesub_filtered[animal]),  plots_folder, todays_datetime)
    pVal_halfSecPreTGB_basesubfilt_byAnimal[animal] = pVal_thisA_halfSecPreTGB_basesubfilt
    sigma_halfSecPreTGB_basesubfilt_byAnimal[animal] = sigma_thisA_halfSecPreTGB_basesubfilt

### POOL ACROSS ALL ANIMALS, from 3.4 seconds after TGB to end
allA_basesub_filt_tb205toEnd_catches, allA_basesub_filt_tb205toEnd_catches_N, allA_basesub_filt_tb205toEnd_misses, allA_basesub_filt_tb205toEnd_misses_N = pool_acrossA_acrossTB(all_catches_basesub_filtered, all_misses_basesub_filtered, 205, -1, "all")
# all animals baseline shuffle test, number of tests = 2
pVal_allA_tb205toEnd_basesubfilt, sigma_allA_tb205toEnd_basesubfilt = shuffle_test(allA_basesub_filt_tb205toEnd_catches, allA_basesub_filt_tb205toEnd_misses, No_of_Shuffles, "AllCatch-BaseSubFilt-tb205toEnd", "AllMiss-BaseSubFilt-tb205toEnd", allA_basesub_filt_tb205toEnd_catches_N, allA_basesub_filt_tb205toEnd_misses_N,  plots_folder, todays_datetime)
### individual animals, from 3.4 seconds after TGB to end (to double check the pooled shuffle test)
postTB205_catches_basesubfilt_byAnimal, postTB205_misses_basesubfilt_byAnimal = pool_timebins_byAnimal(all_catches_basesub_filtered, all_misses_basesub_filtered, 205, -1)
pVal_postTB205_basesubfilt_byAnimal = {}
sigma_postTB205_basesubfilt_byAnimal = {}
for animal in postTB205_catches_basesubfilt_byAnimal:
    pVal_thisA_postTB205_basesubfilt, sigma_thisA_postTB205_basesubfilt = shuffle_test(postTB205_catches_basesubfilt_byAnimal[animal], postTB205_misses_basesubfilt_byAnimal[animal], indivA_N_shuffles, animal+"-BaseSubFilt-halfSecPreTGB", animal+"-BaseSubFilt-halfSecPreTGB", len(all_catches_basesub_filtered[animal]), len(all_misses_basesub_filtered[animal]),  plots_folder, todays_datetime)
    pVal_postTB205_basesubfilt_byAnimal[animal] = pVal_thisA_postTB205_basesubfilt
    sigma_postTB205_basesubfilt_byAnimal[animal] = sigma_thisA_postTB205_basesubfilt

########################################################
### ------ under construction!!!! ------ ###
########################################################

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

f = Fitter(np.hstack(allO_allA_postTGB))
f.fit()
plt.hist(np.hstack(allO_allA_postTGB))
f.summary()
#>>> f.summary()
##            sumsquare_error
##levy             129.317025
##invgauss         239.201950
##invgamma         248.078092
##halfcauchy       344.966041
##expon            360.504135
plt.show()



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