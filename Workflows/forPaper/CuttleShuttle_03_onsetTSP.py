# -*- coding: utf-8 -*-
"""
process_cuttle_python.py

Loads intermediate files generated by process_cuttle_python_01_genBandEnergies.py
Baseline and normalise band energies, pool across animals categorized by catch versus miss
Calculate 3 sigma bounds from mean pooled baseline
Characterise dynamics of the Tentacle Shot Pattern (TSP)

Optional flags:
"--run_type": 'prototype'(default) or 'collab'
"--baseline": 60 (default) or any integer value from 1-179
"--plot_indiv_animals": False (default) or True
"--plot_pooled_animals": False (default) or True
"--plot_pooled_percentchange": False (default) or True
"--plot_baseline_hist": False (default) or True
"--plot_3sigCI": False (default) or True
"--N_freqBands": 4 (default) or any integer value from 1-7, determines how many frequency bands will be included in TSP characterisation
"--smoothing_window": 15 (default), used for parameter "windown_length" in python function scipy.signal.savgol_filter()

@author: Danbee Kim and Adam R Kampff
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import cv2
import datetime
import logging
import pdb
import argparse
import scipy.signal

###################################
# SET CURRENT WORKING DIRECTORY
###################################
cwd = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
today_dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename="process_cuttle_python_02_" + today_dateTime + ".log", filemode='w', level=logging.INFO)
###################################
# FUNCTIONS
###################################
##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) data_folder (parent folder with all intermediate data)
# AND
# 2) plots_folder (parent folder for all plots output from analysis scripts)
### Current default uses a debugging source dataset
##########################################################
def load_data(run_type='prototype'):
    if run_type == 'prototype':
        data_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\data'
        plots_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\draftPlots'
    elif run_type == 'collab':
        data_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\data'
        plots_dir = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\plots'
    return data_dir, plots_dir
##########################################################

def categorize_by_animal(TGB_files):
    all_animals_dict = {}
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files:
        TGB_name = os.path.basename(TGB_file)
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TS_bandEnergies = np.load(TGB_file)
        # extract power at each frequency band for every frame
        all_bands = range(TS_bandEnergies.shape[1])
        power_at_each_frequency = {key:[] for key in all_bands}
        for frame in TS_bandEnergies:
            for band in frame:
                i, = np.where(frame == band)[0]
                power_at_each_frequency[i].append(band)
        all_animals_dict.setdefault(TGB_animal,[]).append(power_at_each_frequency)
    return all_animals_dict

def categorize_by_animal_catchVmiss(TGB_files):
    catch_dict = {}
    miss_dict = {}
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = os.path.basename(TGB_file)
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TS_bandEnergies = np.load(TGB_file)
        # extract power at each frequency band for every frame
        all_bands = range(TS_bandEnergies.shape[1])
        power_at_each_frequency = {key:[] for key in all_bands}
        for frame in TS_bandEnergies:
            for band in frame:
                i, = np.where(frame == band)[0]
                power_at_each_frequency[i].append(band)
        if TGB_type == "catch":
            catch_dict.setdefault(TGB_animal,[]).append(power_at_each_frequency)
        if TGB_type == "miss": 
            miss_dict.setdefault(TGB_animal,[]).append(power_at_each_frequency)
    return catch_dict, miss_dict

def percent_change_from_baseline(TS_dict, prey_type, baseline_len):
    percentChange_TS = {}
    # make baseline for each animal, catch vs miss
    for animal in TS_dict:
        percentChange_TS[animal] = {}
        try:
            # baseline subtract each frequency during each trial
            allFreq_allTrials_percentChange = {}
            for i,trial in enumerate(TS_dict[animal]):
                for freq_band in trial:
                    percentChange_TS[animal][freq_band] = {}
                    this_freq_baseline = np.nanmean(TS_dict[animal][i][freq_band][0:baseline_len])
                    this_freq_percentChange = [(float(x/this_freq_baseline)-1)*100 for x in TS_dict[animal][i][freq_band]]
                    allFreq_allTrials_percentChange.setdefault(freq_band,[]).append(this_freq_percentChange)
            for freq_band in allFreq_allTrials_percentChange:
                thisFreq_baseSub_mean_byFrame = np.nanmean(allFreq_allTrials_percentChange[freq_band], axis=0)
                thisFreq_baseSub_mean_byTrial = np.nanmean(allFreq_allTrials_percentChange[freq_band])
                thisFreq_baseSub_std_byFrame = np.nanstd(allFreq_allTrials_percentChange[freq_band], axis=0, ddof=1)
                thisFreq_baseSub_std_byTrial = np.nanstd(allFreq_allTrials_percentChange[freq_band], ddof=1)
                percentChange_TS[animal][freq_band]['trials'] = allFreq_allTrials_percentChange[freq_band]
                percentChange_TS[animal][freq_band]['mean frame'] = thisFreq_baseSub_mean_byFrame
                percentChange_TS[animal][freq_band]['mean trial'] = thisFreq_baseSub_mean_byTrial
                percentChange_TS[animal][freq_band]['std frame'] = thisFreq_baseSub_std_byFrame
                percentChange_TS[animal][freq_band]['std trial'] = thisFreq_baseSub_std_byTrial
        except Exception:
            print("{a} made no tentacle shots during {p} prey movement type".format(a=animal, p=prey_type))
    return percentChange_TS

def pooled_mean_var_allAnimals(allA_meanPercentChange_dict):
    # calculate mean and variance across all animals
    pooled_means = {}
    pooled_stds = {}
    for freq_band in allA_meanPercentChange_dict['N'].keys():
        # find pooled mean
        pooled_mean_numerator = []
        pooled_denominator = []
        for animal in range(len(allA_meanPercentChange_dict['N'][freq_band])):
            this_animal_mean_numerator = allA_meanPercentChange_dict['N'][freq_band][animal]*allA_meanPercentChange_dict['Mean'][freq_band][animal]
            pooled_mean_numerator.append(this_animal_mean_numerator)
            pooled_denominator.append(allA_meanPercentChange_dict['N'][freq_band][animal])
        this_freq_pooled_mean = np.sum(pooled_mean_numerator, axis=0)/np.sum(pooled_denominator)
        # find pooled variance
        pooled_var_numerator = []
        for animal in range(len(allA_meanPercentChange_dict['N'][freq_band])):
            this_animal_var_numerator = []
            for trial in allA_meanPercentChange_dict['trials'][freq_band][animal]:
                this_trial_var = np.square(trial-this_freq_pooled_mean)
                this_animal_var_numerator.append(this_trial_var)
            pooled_var_numerator.append(np.sum(this_animal_var_numerator, axis=0))
        this_freq_pooled_var = np.sum(pooled_var_numerator, axis=0)/(np.sum(pooled_denominator)-1)
        pooled_means[freq_band] = this_freq_pooled_mean
        pooled_stds[freq_band] = np.sqrt(this_freq_pooled_var)
    return pooled_means, pooled_stds

def collect_across_animals(percent_change_dict, collected_dict, ts_category_str):
    collected_dict[ts_category_str] = {'N': {}, 'Mean': {}, 'trials': {}}
    for animal in percent_change_dict.keys():
        for freq_band in percent_change_dict[animal].keys():
            this_animal_this_freq_N = len(percent_change_dict[animal][freq_band]['trials'])
            this_animal_this_freq_mean = percent_change_dict[animal][freq_band]['mean frame']
            collected_dict[ts_category_str]['N'].setdefault(freq_band,[]).append(this_animal_this_freq_N)
            collected_dict[ts_category_str]['Mean'].setdefault(freq_band,[]).append(this_animal_this_freq_mean)
            collected_dict[ts_category_str]['trials'].setdefault(freq_band,[]).append(percent_change_dict[animal][freq_band]['trials'])

def pool_across_animals(collected_dict, pooled_dict, pooled_by_tb, ts_category_str):
    pooled_dict[ts_category_str] = {'pooled mean': {}, 'pooled N': {}, 'pooled trials': {}}
    for freq_band in collected_dict[ts_category_str]['trials']:
        pooled_trials = []
        for animal in range(len(collected_dict[ts_category_str]['trials'][freq_band])):
            for trial in collected_dict[ts_category_str]['trials'][freq_band][animal]:
                pooled_trials.append(trial)
                for timebucket, percent_change in enumerate(trial):
                    pooled_by_tb[ts_category_str].setdefault(freq_band,{}).setdefault(timebucket,[]).append(percent_change)
        pooled_N_this_fb = sum(collected_dict[ts_category_str]['N'][freq_band])
        mean_this_fb = np.nanmean(pooled_trials, axis=0)
        pooled_dict[ts_category_str]['pooled mean'][freq_band] = mean_this_fb
        pooled_dict[ts_category_str]['pooled N'][freq_band] = pooled_N_this_fb
        pooled_dict[ts_category_str]['pooled trials'][freq_band] = pooled_trials

def smooth_pooled_trials(pooled_dict, smoothing_window_tbs, smoothed_trials_dict, N_of_freq_bands_to_smooth):
    for ts_type in pooled_dict:
        for freq_band in range(N_of_freq_bands_to_smooth):
            for t, trial in enumerate(pooled_dict[ts_type]['pooled trials'][freq_band]):
                try:
                    smoothed_trial = scipy.signal.savgol_filter(trial, smoothing_window, 3)
                    smoothed_trials_dict[ts_type].setdefault(freq_band,[]).append(smoothed_trial)
                except:
                    print('Trial {t} failed'.format(t=t))

def plot_BaselineHistograms_perFreqBand(analysis_type_str, preprocess_str, metric_str, prey_type_str, observed_baseline_dict, freq_band, baseline_len, todays_dt, plots_dir):
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_pooledAnimals_FreqBand'+str(freq_band)+'_baselineHistSanityCheck_'+todays_dt+'.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Histogram of baseline period of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Frequency Band {fb} \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Prey Movement type: {p}, pooled across all animals'.format(m=metric_str, at=analysis_type_str, fb=str(freq_band), b=str(baseline_len/60), p=prey_type_str)
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.hist(observed_baseline_dict[freq_band], bins=90, normed=True)
    # visual check to see if the distribution is gaussian
    mean_baseline = np.nanmean(observed_baseline_dict[freq_band])
    std_baseline = np.nanstd(observed_baseline_dict[freq_band])
    x = np.linspace(min(observed_baseline_dict[freq_band]), max(observed_baseline_dict[freq_band]), 100)
    f = np.exp(-(1/2)*np.power((x - mean_baseline)/std_baseline,2)) / (std_baseline*np.sqrt(2*np.pi))
    plt.plot(x,f, label='gaussian distribution')
    plt.legend()
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_3sigCI_individualTS_per_FreqBand(analysis_type_str, preprocess_str, metric_str, prey_type_str, ts_category_str, freq_band, pooled_trials_this_fb, baseline_stats_dict, baseline_len, TGB_bucket):
    N_TS = len(pooled_trials_this_fb)
    # set colors
    color_individualTS = [0.8431, 0.1882, 0.1529, 0.05]
    color_baseline = [0.0, 0.53333, 0.215686, 1.0]
    color_baseline_3sigma = [0.0, 0.533, 0.2157, 0.1]
    color_TGB = [0.4627, 0.1647, 0.5137, 1.0]
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_pooledAnimals_'+ts_category_str+'_FreqBand'+str(freq_band)+'_3sigCI_'+today_dateTime+'.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = 'Distribution of percent change from mean baseline of {m} in ROI on cuttlefish mantle during {ts_cat} tentacle shots, as detected by {at}\n Frequency Band {fb} \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}'.format(m=metric_str, ts_cat=ts_category_str, at=analysis_type_str, fb=str(freq_band), b=str(baseline_frames/60), p=prey_type_str, Nts=str(N_TS))
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plot_xticks = np.arange(0, 360, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    for trial in pooled_trials_this_fb:
        plt.plot(trial, color=color_individualTS)
    mean_baseline = baseline_stats_dict['mean'][freq_band]
    baseline_3sigCI = baseline_stats_dict['std'][freq_band]*3
    upper_bound = mean_baseline+baseline_3sigCI
    lower_bound = mean_baseline-baseline_3sigCI
    plt.fill_between(range(360), upper_bound, lower_bound, color=color_baseline_3sigma)
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), linestyle='--', linewidth=2, color=color_baseline)
    plt.text(baseline_len, ymax, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor=color_baseline, boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), linestyle='--', linewidth=2, color=color_TGB)
    plt.text(TGB_bucket, ymax, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor=color_TGB, boxstyle='round,pad=0.35'))
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def TSP_detector(pooled_dict, ts_category_str, plot_3sigCI, real_exit_window_tbs, onsets_dict, onsets_yScatter, offsets_dict, baseline_stats_dict, baseline_len, TGB_bucket):
    for freq_band in pooled_dict[ts_category_str]:
        all_trials_this_freq_band = pooled_dict[ts_category_str][freq_band]
        #N_trials = len(all_trials_this_freq_band)
        #print('Number of trials: {n}'.format(n=str(N_trials)))
        # visualize distribution of onset of tentacle shot pattern
        if plot_3sigCI:
            plot_3sigCI_individualTS_per_FreqBand('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'all', ts_category_str, freq_band, all_trials_this_freq_band, baseline_stats_dict, baseline_len, TGB_bucket)
        # numerically calculate when each individual trace leaves the 3sigCI
        this_fb_mean = baseline_stats_dict['mean'][freq_band]
        this_fb_3sig = baseline_stats_dict['std'][freq_band]*3
        this_fb_3sigCI_upper = this_fb_mean + this_fb_3sig
        this_fb_3sigCI_lower = this_fb_mean - this_fb_3sig
        this_fb_onsets = []
        this_fb_y = []
        this_fb_offsets = []
        for t, trial in enumerate(all_trials_this_freq_band):
            onset_candidate = None
            best_onset = None
            shortest_time_from_TGB = None
            best_offset = None
            ### FIND EXIT THAT IS CLOSEST (ABS VALUE) TO TGB ###
            for i, timebucket in enumerate(trial):
                if onset_candidate is None:
                    if freq_band==0:
                        if timebucket<this_fb_3sigCI_lower:
                            onset_candidate = i
                            #print('Onset candidate at timebucket {i}...'.format(i=i))
                            continue
                    else:
                        if timebucket>this_fb_3sigCI_upper:
                            onset_candidate = i
                            #print('Onset candidate at timebucket {i}...'.format(i=i))
                            continue
                if onset_candidate is not None:
                    if this_fb_3sigCI_lower<timebucket<this_fb_3sigCI_upper:
                        #print('Re-entered at frame {i}...'.format(i=i))
                        if i>TGB_bucket:
                            time_from_TGB = abs(onset_candidate-TGB_bucket)
                            #print('Current time from TGB = {t}, Shortest time from TGB = {st}'.format(t=time_from_TGB, st=shortest_time_from_TGB))
                            if shortest_time_from_TGB is None:
                                best_onset = onset_candidate
                                best_offset = i
                                shortest_time_from_TGB = time_from_TGB
                                onset_candidate = None
                                #print('First best offset at timebucket {off}, best onset at timebucket {on}, shortest time from TGB = {st}...'.format(off=best_offset, on=best_onset, st=shortest_time_from_TGB))
                                continue
                            elif time_from_TGB<=shortest_time_from_TGB:
                                best_onset = onset_candidate
                                best_offset = i
                                shortest_time_from_TGB = time_from_TGB
                                onset_candidate = None
                                #print('New best offset at timebucket {off}, best onset at timebucket {on}, shortest time from TGB = {st}...'.format(off=best_offset, on=best_onset, st=shortest_time_from_TGB))
                                continue
                            elif time_from_TGB>shortest_time_from_TGB:
                                onset_candidate = None
                                #print('Current best offset at timebucket {off}, best onset at timebucket {on}, shortest time from TGB = {st}...'.format(off=best_offset, on=best_onset, st=shortest_time_from_TGB))
                                continue
                        else:
                            onset_candidate = None
                            continue
                if i==len(trial)-1:
                    if best_offset is None:
                        if onset_candidate is not None and best_onset is None:
                            this_fb_onsets.append(onset_candidate)
                            this_fb_y.append(freq_band)
                            this_fb_offsets.append(np.nan)
                            #print('FOUND first real exit for trial {t} at timebucket {on}, with no re-entry by end of trial'.format(t=t, on=onset_candidate))
                        if onset_candidate is not None and best_onset is not None:
                            this_fb_onsets.append(best_onset)
                            this_fb_y.append(freq_band)
                            this_fb_offsets.append(np.nan)
                            #print('FOUND first real exit for trial {t} at timebucket {on}, with no re-entry by end of trial'.format(t=t, on=best_onset)) 
                    else:
                        this_fb_onsets.append(best_onset)
                        this_fb_y.append(freq_band)
                        this_fb_offsets.append(best_offset)
                        #print('FOUND first real exit for trial {t} at timebucket {on}, with re-entry at timebucket {off}'.format(t=t, on=best_onset, off=best_offset))
        #print('Number of first exits found: {N}'.format(N=len(this_fb_onsets)))
        onsets_dict[ts_category_str].append(this_fb_onsets)
        onsets_yScatter[ts_category_str].append(this_fb_y)
        offsets_dict[ts_category_str].append(this_fb_offsets)

def plot_TSPdynamics_hist_perFreqBand(analysis_type_str, preprocess_str, metric_str, tentacle_shot_type, onsets_dict, first_reEntries_dict, real_exit_window_tbs, freq_band, baseline_len, todays_dt, plots_dir):  
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_pooledAnimals_'+tentacle_shot_type+'_FreqBand'+str(freq_band)+'_TSP-firstAppearance_window'+str(real_exit_window_tbs)+'tbs_'+todays_dt+'.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Histogram of first frame when {m} in ROI on cuttlefish mantle is greater than 3 sigma away from baseline mean for at least {w} milliseconds \n As detected by {at}, Frequency Band {fb} \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Tentacle Shot type: {ts}, pooled across all animals'.format(m=metric_str, w=str(1/60*real_exit_window_tbs), at=analysis_type_str, fb=str(freq_band), b=str(baseline_len/60), ts=tentacle_shot_type)
    # setup fig
    plt.figure(figsize=(16,16), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    # subplot: appearance of TSP
    plt.subplot(2,1,1)
    plt.title('Timing of first appearance of TSP relative to TGB (3 seconds)', fontsize=10, color='grey', style='italic')
    plt.xlabel("Seconds")
    plot_xticks = np.arange(0, 360, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.hist(onsets_dict[tentacle_shot_type][freq_band], bins=90, normed=True)
    # visual check to see if the distribution is gaussian
    mean_exit = np.nanmean(onsets_dict[tentacle_shot_type][freq_band])
    std_exit = np.nanstd(onsets_dict[tentacle_shot_type][freq_band])
    x = np.linspace(min(onsets_dict[tentacle_shot_type][freq_band]), max(onsets_dict[tentacle_shot_type][freq_band]), 1000)
    f = np.exp(-(1/2)*np.power((x - mean_exit)/std_exit,2)) / (std_exit*np.sqrt(2*np.pi))
    plt.plot(x,f, label='gaussian distribution')
    plt.legend()
    # subplot: disappearance of TSP
    plt.subplot(2,1,2)
    plt.title('Timing of disappearance of TSP relative to TGB (3 seconds)', fontsize=10, color='grey', style='italic')
    plt.xlabel("Seconds")
    plot_xticks = np.arange(0, 360, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.hist(first_reEntries_dict[tentacle_shot_type][freq_band], bins=90, normed=True)
    # visual check to see if the distribution is gaussian
    mean_exit = np.nanmean(first_reEntries_dict[tentacle_shot_type][freq_band])
    std_exit = np.nanstd(first_reEntries_dict[tentacle_shot_type][freq_band])
    x = np.linspace(min(first_reEntries_dict[tentacle_shot_type][freq_band]), max(first_reEntries_dict[tentacle_shot_type][freq_band]), 1000)
    f = np.exp(-(1/2)*np.power((x - mean_exit)/std_exit,2)) / (std_exit*np.sqrt(2*np.pi))
    plt.plot(x,f, label='gaussian distribution')
    plt.legend()
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def boxplots_of_TSP_onset(analysis_type_str, preprocess_str, metric_str, ts_category_str, onsets_dict, y_scatter_dict, onset_stats_dict, N_fb_to_plot, baseline_len, TGB_bucket, todays_dt, plots_dir):
    N_TS_str = 'Number of tentacle shots: '
    for f, freq_band in enumerate(range(N_fb_to_plot)):
        N_TS_thisFB = len(onsets_dict[ts_category_str][freq_band])
        if f == N_fb_to_plot:
            N_TS_str = N_TS_str+'{n} (Freq Band {f})'.format(n=N_TS_thisFB, f=f)
        else:
            N_TS_str = N_TS_str+'{n} (Freq Band {f}), '.format(n=N_TS_thisFB, f=f)
    mean_onset_str = 'Mean onset of TSP (seconds relative to TGB): '
    for i, m_onset in enumerate(onset_stats_dict[ts_category_str]['mean']):
        std = onset_stats_dict[ts_category_str]['std'][i]
        if i == len(onset_stats_dict[ts_category_str]['mean']):
            mean_onset_str = mean_onset_str+'{m:.3f}+-{std:.3f} (Freq Band {i})'.format(m=(m_onset/60)-3, std=(std/60), i=i)
        else:
            mean_onset_str = mean_onset_str+'{m:.3f}+-{std:.3f} (Freq Band {i}), '.format(m=(m_onset/60), std=(std/60), i=i)
    # set colors
    color_baseline = [0.0, 0.53333, 0.215686, 1.0]
    color_TGB = [0.4627, 0.1647, 0.5137, 1.0]
    color_catch = [0.2706, 0.4588, 0.70588, 1.0]
    color_miss = [0.8431, 0.1882, 0.1529, 1.0]
    color_meanline = [0.9137, 0.470588, 0.1529, 1.0]
    # set properties for boxplots
    medianprops = dict(linewidth=0)
    meanlineprops = dict(linestyle='-', linewidth=3, color=color_meanline)
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_onsetTSP_'+ts_category_str+'TS_FreqBand0-'+str(N_fb_to_plot)+todays_dt+'.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Boxplots showing distribution of TSP onset, as detected by {at}, Frequency Bands 0 to {fb} during {ts_cat} tentacle shots \n Pooled across all animals, {Nts} \n {MOstr}'.format(ts_cat=ts_category_str, at=analysis_type_str, fb=str(N_fb_to_plot-1), Nts=N_TS_str, MOstr=mean_onset_str)
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    # setup points to plot
    first_exits_toPlot = onsets_dict[ts_category_str][:N_fb_to_plot]
    y_scatter_toPlot = y_scatter_dict[ts_category_str][:N_fb_to_plot]
    plt.boxplot(first_exits_toPlot, vert=False, meanline=True, showmeans=True, medianprops=medianprops, meanprops=meanlineprops)
    for freq_band in range(len(first_exits_toPlot)):
        if ts_category_str == 'all':
            first_exits_plot_catch = onsets_dict['catches'][:N_fb_to_plot]
            first_exits_plot_miss = onsets_dict['misses'][:N_fb_to_plot]
            y_scatter_plot_catch = y_scatter_dict['catches'][:N_fb_to_plot]
            y_scatter_plot_miss = y_scatter_dict['misses'][:N_fb_to_plot]
            jitter_for_plotting_catch = np.random.normal([x+1 for x in y_scatter_plot_catch[freq_band]], 0.08, size=len(first_exits_plot_catch[freq_band]))
            jitter_for_plotting_miss = np.random.normal([x+1 for x in y_scatter_plot_miss[freq_band]], 0.08, size=len(first_exits_plot_miss[freq_band]))
            plt.plot(first_exits_plot_catch[freq_band], jitter_for_plotting_catch, '.', color=color_catch)
            plt.plot(first_exits_plot_miss[freq_band], jitter_for_plotting_miss, '.', color=color_miss)
        else:
            if ts_category_str == 'catches':
                color_to_plot = color_catch
            elif ts_category_str == 'misses':
                color_to_plot = color_miss
            jitter_for_plotting = np.random.normal([x+1 for x in y_scatter_toPlot[freq_band]], 0.08, size=len(first_exits_toPlot[freq_band]))
            plt.plot(first_exits_toPlot[freq_band], jitter_for_plotting, '.', color=color_to_plot)
    # adjust axes labels
    plt.ylabel("Frequency bands")
    plt.ylim(0.5,N_fb_to_plot+0.5)
    plot_yticks = np.arange(1, N_fb_to_plot+1, 1)
    plt.yticks(plot_yticks, ['%d'%(y-1) for y in plot_yticks])
    plt.xlabel("Seconds")
    plt.xlim(-10,360)
    plot_xticks = np.arange(0, 360, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), linestyle='--', linewidth=2, color=color_baseline)
    #plt.text(baseline_len, ymax-0.15, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor=color_baseline, boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), linestyle='--', linewidth=2, color=color_TGB)
    #plt.text(TGB_bucket, ymax-0.15, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor=color_TGB, boxstyle='round,pad=0.35'))
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--run_type", nargs='?', default='prototype')
    parser.add_argument("--baseline", nargs='?', default=60)
    parser.add_argument("--plot_indiv_animals", nargs='?', default=False)
    parser.add_argument("--plot_pooled_animals", nargs='?', default=False)
    parser.add_argument("--plot_pooled_percentchange", nargs='?', default=False)
    parser.add_argument("--plot_baseline_hist", nargs='?', default=False)
    parser.add_argument("--plot_3sigCI", nargs='?', default=False)
    parser.add_argument("--N_freqBands", nargs='?', default=4)
    parser.add_argument("--smoothing_window", nargs='?', default=15)
    args = parser.parse_args()
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    data_folder, plots_folder = load_data(args.run_type)
    logging.info('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (data_folder, plots_folder))
    print('DATA FOLDER: %s \n PLOTS FOLDER: %s' % (data_folder, plots_folder))
    ###################################
    # PLOT TOGGLES
    ###################################
    plot_indiv_animals = args.plot_indiv_animals
    plot_pooled_animals = args.plot_pooled_animals
    plot_pooled_percentchange = args.plot_pooled_percentchange
    plot_baseline_hist = args.plot_baseline_hist
    plot_3sigCI = args.plot_3sigCI
    plot_TSP_dynamics_hist = args.plot_TSP_dynamics_hist
    N_freqBands = args.N_freqBands
    smoothing_window = args.smoothing_window
    ###################################
    # COLLECT DATA FROM DATA_FOLDER
    ###################################
    # collect all binary files with granularity data
    all_data = glob.glob(data_folder + os.sep + "*.npy")
    ########################################################
    ### ------ ORGANIZE DATA ------ ###
    ########################################################
    # categorize tentacle shots according to prey movement
    TGB_natural = []
    TGB_patterned = []
    TGB_causal = []
    TGB_daily = {}
    for TGB_file in all_data:
        trial_date = os.path.basename(TGB_file).split('_')[2]
        sorted_by_session = TGB_daily.setdefault(trial_date,[]).append(TGB_file)
        trial_datetime = datetime.datetime.strptime(trial_date, '%Y-%m-%d')
        if trial_datetime < datetime.datetime(2014, 9, 13, 0, 0):
            TGB_natural.append(TGB_file)
        elif trial_datetime > datetime.datetime(2014, 10, 18, 0, 0):
            TGB_causal.append(TGB_file)
        else:
            TGB_patterned.append(TGB_file)
    # organize power-at-frequency-band data
    # categorize daily sessions by animal
    all_TS_daily = {}
    all_catches_daily = {}
    all_misses_daily = {}
    for session_date in TGB_daily:
        all_TS_daily[session_date] = categorize_by_animal(TGB_daily[session_date])
    # collect all power-at-frequency-band data and categorize by animal
    all_TS = categorize_by_animal(all_data)
    # collect all power-at-frequency-band data and categorize by animal and type (catch vs miss)
    all_catches, all_misses = categorize_by_animal_catchVmiss(all_data)
    all_raw = [all_catches, all_misses]
    # frame for moment tentacles go ballistic
    TGB_bucket_raw = 180
    ########################################################
    ### ------ DATA NORMALIZATION/STANDARDIZATION ------ ###
    ########################################################
    baseline_frames = args.baseline
    # convert power at frequency into percent change from baseline
    # all tentacle shots
    dailyTS_percentChange = {}
    for session_date in all_TS_daily:
        dailyTS_percentChange[session_date] = percent_change_from_baseline(all_TS_daily[session_date], 'all', baseline_frames)
    allTS_percentChange = percent_change_from_baseline(all_TS, 'all', baseline_frames)
    # all catches
    dailyCatches_percentChange = {}
    for session_date in all_catches_daily:
        dailyCatches_percentChange[session_date] = percent_change_from_baseline(all_catches_daily[session_date], 'all', baseline_frames)
    allCatches_percentChange = percent_change_from_baseline(all_catches, 'all', baseline_frames)
    # all misses
    dailyMisses_percentChange = {}
    for session_date in all_misses_daily:
        dailyMisses_percentChange[session_date] = percent_change_from_baseline(all_misses_daily[session_date], 'all', baseline_frames)
    allMisses_percentChange = percent_change_from_baseline(all_misses, 'all', baseline_frames)
    #########################################
    ### ------ COLLECT ACROSS ANIMALS ------ ###
    #########################################
    # pool across all animals to plot mean percent change in each frequency for all animals
    percentChange_allAnimals = {'all': {}, 'catches': {}, 'misses': {}}
    collect_across_animals(allTS_percentChange, percentChange_allAnimals, 'all')
    collect_across_animals(allCatches_percentChange, percentChange_allAnimals, 'catches')
    collect_across_animals(allMisses_percentChange, percentChange_allAnimals, 'misses')
    #######################################################
    ### ------ ONSET OF SIG CHANGE FROM BASELINE ------ ###
    #######################################################
    # create pools of all tentacle shots for each freq band
    percentChange_pooledAnimals = {'all': {}, 'catches': {}, 'misses': {}}
    percentChange_pooled_by_TB = {'all': {}, 'catches': {}, 'misses': {}}
    pool_across_animals(percentChange_allAnimals, percentChange_pooledAnimals, percentChange_pooled_by_TB, 'all')
    pool_across_animals(percentChange_allAnimals, percentChange_pooledAnimals, percentChange_pooled_by_TB, 'catches')
    pool_across_animals(percentChange_allAnimals, percentChange_pooledAnimals, percentChange_pooled_by_TB, 'misses')
    # calculate distribution of values during baseline from all tentacle shots
    pool_of_observed_baseline_values = {}
    for freq_band in percentChange_allAnimals['all']['trials']:
        for animal in range(len(percentChange_allAnimals['all']['trials'][freq_band])):
            for trial in percentChange_allAnimals['all']['trials'][freq_band][animal]:
                this_trial_baseline = trial[:baseline_frames]
                for value in this_trial_baseline:
                    pool_of_observed_baseline_values.setdefault(freq_band,[]).append(value)
    # establish stats for baseline
    baseline_stats = {'mean': {}, 'std': {}}
    for freq_band in pool_of_observed_baseline_values:
        if plot_baseline_hist:
            # sanity check the distribution of the baseline values, is it close enough to gaussian?
            plot_BaselineHistograms_perFreqBand('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'all', pool_of_observed_baseline_values, freq_band, baseline_frames, today_dateTime, plots_folder)
        mean_baseline_this_freq = np.nanmean(pool_of_observed_baseline_values[freq_band])
        std_baseline_this_freq = np.nanstd(pool_of_observed_baseline_values[freq_band])
        baseline_stats['mean'][freq_band] = mean_baseline_this_freq
        baseline_stats['std'][freq_band] = std_baseline_this_freq
    ########################################################
    ### ------ TSP DETECTION AND CHARACTERISATION ------ ###
    ########################################################
    # smooth individual trials for easier TSP detection
    smoothed_pooledAnimals = {'all': {}, 'catches': {}, 'misses': {}}
    smooth_pooled_trials(percentChange_pooledAnimals, smoothing_window, smoothed_pooledAnimals, N_freqBands)
    # visually and numerically check when mantle pattern deviates significantly from baseline
    for freq_band in range(3):
        lower_bound = baseline_stats['mean'][freq_band] - 3*baseline_stats['std'][freq_band]
        upper_bound = baseline_stats['mean'][freq_band] + 3*baseline_stats['std'][freq_band]
        print('Freq band: {fb}, lower bound: {lb}, upper bound: {ub}'.format(fb=freq_band, lb=lower_bound, ub = upper_bound))
    TSP_onsets_3sigCI = {'all': [], 'catches': [], 'misses': []}
    TSP_onsets_y_scatter = {'all': [], 'catches': [], 'misses': []}
    TSP_offsets_3sigCI = {'all': [], 'catches': [], 'misses': []}
    real_exit_window = 15 #frames/timebuckets
    TSP_detector(smoothed_pooledAnimals, 'all', plot_3sigCI, real_exit_window, TSP_onsets_3sigCI, TSP_onsets_y_scatter, TSP_offsets_3sigCI, baseline_stats, baseline_frames, TGB_bucket_raw)
    TSP_detector(smoothed_pooledAnimals, 'catches', plot_3sigCI, real_exit_window, TSP_onsets_3sigCI, TSP_onsets_y_scatter, TSP_offsets_3sigCI, baseline_stats, baseline_frames, TGB_bucket_raw)
    TSP_detector(smoothed_pooledAnimals, 'misses', plot_3sigCI, real_exit_window, TSP_onsets_3sigCI, TSP_onsets_y_scatter, TSP_offsets_3sigCI, baseline_stats, baseline_frames, TGB_bucket_raw)
    # check distribution of first exits
    if plot_TSP_dynamics_hist:
        for freq_band in range(len(TSP_onsets_3sigCI['all'])):
            plot_TSPdynamics_hist_perFreqBand('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'all', TSP_onsets_3sigCI, TSP_offsets_3sigCI, 15, freq_band, baseline_frames, today_dateTime, plots_folder)
        for freq_band in range(len(TSP_onsets_3sigCI['catches'])):
            plot_TSPdynamics_hist_perFreqBand('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'catches', TSP_onsets_3sigCI, TSP_offsets_3sigCI, 15, freq_band, baseline_frames, today_dateTime, plots_folder)
        for freq_band in range(len(TSP_onsets_3sigCI['misses'])):
            plot_TSPdynamics_hist_perFreqBand('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'misses', TSP_onsets_3sigCI, TSP_offsets_3sigCI, 15, freq_band, baseline_frames, today_dateTime, plots_folder)
    # find mean onset, std of onset
    onset_TSP = {'all': {}, 'catches': {}, 'misses': {}}
    for ts_type in onset_TSP:
        onset_TSP[ts_type] = {'mean': [], 'std': []}
        for freq_band in range(len(TSP_onsets_3sigCI['all'])):
            onset_TSP[ts_type]['mean'].append(np.nanmean(TSP_onsets_3sigCI[ts_type][freq_band]))
            onset_TSP[ts_type]['std'].append(np.nanstd(TSP_onsets_3sigCI[ts_type][freq_band]))
    ####################################################
    ### ------ PLOT DISTRIBUTION OF TSP ONSET ------ ###
    ####################################################
    # make boxplots to show distribution of "onset of tentacle shot pattern"
    boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'all', TSP_onsets_3sigCI, TSP_onsets_y_scatter, onset_TSP, N_freqBands, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)
    boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'catches', TSP_onsets_3sigCI, TSP_onsets_y_scatter, onset_TSP, N_freqBands, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)
    boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'misses', TSP_onsets_3sigCI, TSP_onsets_y_scatter, onset_TSP, N_freqBands, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)

# FIN