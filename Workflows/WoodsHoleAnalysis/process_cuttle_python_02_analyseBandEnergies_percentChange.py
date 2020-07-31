# -*- coding: utf-8 -*-
"""
process_cuttle_python.py

Loads intermediate files generated by process_cuttle_python_01_genBandEnergies.py
Baseline and normalise band energies, pool across animals categorized by catch versus miss
Make a shuffle test of the data and plot

Optional flags:
"--run_type": 'prototype' (default) or 'collab'
"--plotZScore": False (default) or True
"--plotRandomTraces": False (default) or True
"--plotShuffles": False (default) or True

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

def plot_percentChange_pooled_animals_allFreq(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_meanPercentChange_dict, TGB_bucket, baseline_len, plots_dir, todays_dt):
    img_type = ['.png', '.pdf']
    # calculate total number of tentacle shots
    N_TS = 0
    for animal in allA_meanPercentChange_dict['N'][0]:
        N_TS += animal
    pooled_means, pooled_stds = pooled_mean_var_allAnimals(allA_meanPercentChange_dict)
    # set fig path and title
    if len(prey_type_str.split(' '))>1:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_allFreqBand_'+prey_type_str.split(' ')[1]+'Trials_'+todays_dt+img_type[0]
    else:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_allFreqBand_'+prey_type_str+'Trials_'+todays_dt+img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Mean percent change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Transparent regions show standard deviation \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, Nts=str(N_TS))
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.ylabel("Percent change from baseline in power")
    plot_xticks = np.arange(0, len(allA_meanPercentChange_dict['Mean'][0][0]), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    #plt.xlim(0,180)
    plt.ylim(-150, 150)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    N_freq_bands = len(allA_meanPercentChange_dict['N'].keys())
    colors = pl.cm.jet(np.linspace(0,1,N_freq_bands))
    for freq_band in pooled_means.keys():
        x_frames = range(360)
        upper_std = pooled_means[freq_band] + pooled_stds[freq_band]
        lower_std = pooled_means[freq_band] - pooled_stds[freq_band]
        plt.fill_between(x_frames, upper_std, lower_std, color=colors[freq_band], alpha=0.03)
        plt.plot(pooled_means[freq_band], linewidth=2, color=colors[freq_band], alpha=0.5, label='Freq Band {fb}'.format(fb=freq_band))
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-50, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-25, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_percentChange_pooled_animals_someFreq(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_meanPercentChange_dict, list_of_freqs_to_plot, TGB_bucket, baseline_len, plots_dir, todays_dt):
    img_type = ['.png', '.pdf']
    # calculate total number of tentacle shots
    N_TS = 0
    for animal in allA_meanPercentChange_dict['N'][0]:
        N_TS += animal
    pooled_means, pooled_stds = pooled_mean_var_allAnimals(allA_meanPercentChange_dict)
    # set fig path and title
    freq_bands_str = 'freqBands'
    for index in range(len(list_of_freqs_to_plot)):
        if index == len(list_of_freqs_to_plot)-1:
            freq_bands_str += str(list_of_freqs_to_plot[index])
        else:
            freq_bands_str += str(list_of_freqs_to_plot[index])+'-'
    if len(prey_type_str.split(' '))>1:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_'+freq_bands_str+'_'+prey_type_str.split(' ')[1]+'Trials_'+todays_dt+img_type[0]
    else:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_'+freq_bands_str+'_'+prey_type_str+'Trials_'+todays_dt+img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Mean percent change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Transparent regions show standard deviation \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}, showing frequency bands {fb}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, Nts=str(N_TS), fb=freq_bands_str[9:])
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.ylabel("Percent change from baseline in power")
    plot_xticks = np.arange(0, len(allA_meanPercentChange_dict['Mean'][0][0]), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    #plt.xlim(0,180)
    plt.ylim(-100, 300)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    N_freq_bands = len(allA_meanPercentChange_dict['N'].keys())
    colors = pl.cm.jet(np.linspace(0,1,N_freq_bands))
    for freq_band in list_of_freqs_to_plot:
        for animal in allA_meanPercentChange_dict['trials'][freq_band]:
            for trial in animal:
                plt.plot(trial, linewidth=1, color=colors[freq_band], alpha=0.03)
        x_frames = range(360)
        upper_std = pooled_means[freq_band] + pooled_stds[freq_band]
        lower_std = pooled_means[freq_band] - pooled_stds[freq_band]
        plt.fill_between(x_frames, upper_std, lower_std, color=colors[freq_band], alpha=0.05)
        plt.plot(pooled_means[freq_band], linewidth=2, color=colors[freq_band], alpha=0.5, label='Freq Band {fb}'.format(fb=freq_band))
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-50, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-25, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

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

def plot_BaselineHistograms_perFreqBand(analysis_type_str, preprocess_str, metric_str, prey_type_str, observed_baseline_dict, freq_band, baseline_len, todays_dt, plots_dir):
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_pooledAnimals_FreqBand'+str(freq_band)+'_baselineHistSanityCheck_'+todays_dt+'.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Histogram of baseline period of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Frequency Band {fb} \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Prey Movement type: {p}, pooled across all animals'.format(m=metric_str, at=analysis_type_str, fb=str(freq_band), b=str(baseline_len/60), p=prey_type_str)
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.hist(observed_baseline_dict[freq_band], bins=140, normed=True)
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
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_pooledAnimals_'+ts_category_str+'_FreqBand'+str(freq_band)+'_3sigCI_'+today_dateTime+'.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = 'Distribution of percent change from mean baseline of {m} in ROI on cuttlefish mantle during {ts_cat} tentacle shots, as detected by {at}\n Frequency Band {fb} \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}'.format(m=metric_str, ts_cat=ts_category_str, at=analysis_type_str, fb=str(freq_band), b=str(baseline_frames/60), p=prey_type_str, Nts=str(N_TS))
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    for trial in pooled_trials_this_fb:
        plt.plot(trial, color='b', alpha=0.05)
    mean_baseline = baseline_stats_dict['mean'][freq_band]
    baseline_3sigCI = baseline_stats_dict['std'][freq_band]*3
    upper_bound = mean_baseline+baseline_3sigCI
    lower_bound = mean_baseline-baseline_3sigCI
    plt.fill_between(range(360), upper_bound, lower_bound, color='r', alpha=0.25)
    plt.plot(mean_baseline, linewidth=2, color='r')
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def find_onset_TSP(pooled_dict, ts_category_str, plot_3sigCI, first_exits_dict, first_exits_yScatter, baseline_stats_dict, baseline_len, TGB_bucket):
    for freq_band in baseline_stats_dict['mean']:
        all_trials_this_freq_band = pooled_dict[ts_category_str]['pooled trials'][freq_band]
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
        this_fb_earliest_exits = []
        this_fb_y = []
        for t, trial in enumerate(all_trials_this_freq_band):
            for i, timebucket in enumerate(trial):
                if timebucket>this_fb_3sigCI_upper or timebucket<this_fb_3sigCI_lower:
                    this_fb_earliest_exits.append(i)
                    this_fb_y.append(freq_band+1)
                    #print('Found first exit for trial {t} at timebucket {i}'.format(t=t, i=i))
                    break
        #print('Number of first exits found: {N}'.format(N=len(this_fb_earliest_exits)))
        first_exits_dict[ts_category_str].append(this_fb_earliest_exits)
        first_exits_yScatter[ts_category_str].append(this_fb_y)

def boxplots_of_TSP_onset(analysis_type_str, preprocess_str, metric_str, ts_category_str, first_exits_list, y_scatter_list, N_fb_to_plot, baseline_len, TGB_bucket, todays_dt, plots_dir):
    N_TS = len(first_exits_list[ts_category_str][0])
    # set fig path and title
    figure_name = analysis_type_str+'_'+preprocess_str+'_onsetTSP_'+ts_category_str+'TS_FreqBand0-'+str(N_fb_to_plot)+todays_dt+'.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Boxplots showing distribution of when {m} in ROI on cuttlefish mantle first deviates significantly from baseline, as detected by {at}\n Frequency Bands 0 to {fb} during {ts_cat} tentacle shots \n Baseline: mean of {m} from t=0 to t={b} second(s) for each trial \n Pooled across all animals, Number of tentacle shots: {Nts}'.format(m=metric_str, ts_cat=ts_category_str, at=analysis_type_str, fb=str(N_fb_to_plot-1), b=str(baseline_len/60), Nts=str(N_TS))
    # setup fig
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    first_exits_toPlot = first_exits_list[ts_category_str][:N_fb_to_plot]
    y_scatter_list_toPlot = y_scatter_list[ts_category_str][:N_fb_to_plot]
    plt.boxplot(first_exits_toPlot, vert=False)
    for freq_band in range(len(first_exits_toPlot)):
        jitter_for_plotting = np.random.normal(y_scatter_list_toPlot[freq_band], 0.08, size=len(first_exits_toPlot[freq_band]))
        plt.plot(first_exits_toPlot[freq_band], jitter_for_plotting, 'g.')
    # adjust axes labels
    plt.ylabel("Frequency bands")
    plt.ylim(0,N_fb_to_plot+1)
    plot_yticks = np.arange(0, N_fb_to_plot+1, 1)
    plt.yticks(plot_yticks, ['%d'%(y-1) for y in plot_yticks])
    plt.xlabel("Seconds")
    plt.xlim(-10,360)
    plot_xticks = np.arange(0, 360, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-0.25, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'r--', linewidth=1)
    plt.text(TGB_bucket, ymax-0.25, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.35'))
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS
###################################
data_folder = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\data'
plots_folder = r'C:\Users\taunsquared\Dropbox\CuttleShuttle\analysis\WoodsHoleAnalysis\draftPlots'
###################################
# PLOT TOGGLES
###################################
plot_indiv_animals = False
plot_pooled_animals = False
plot_pooled_percentchange = False
plot_baseline_hist = False
plot_3sigCI = False
###################################
# COLLECT DATA FROM DATA_FOLDER
###################################
# collect all binary files with power-at-freq-band data
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
baseline_frames = 60 #frames
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
# visually and numerically check when mantle pattern deviates significantly from baseline
first_exits_3sigCI = {'all': [], 'catches': [], 'misses': []}
first_exits_y_scatter = {'all': [], 'catches': [], 'misses': []}
find_onset_TSP(percentChange_pooledAnimals, 'all', plot_3sigCI, first_exits_3sigCI, first_exits_y_scatter, baseline_stats, baseline_frames, TGB_bucket_raw)
find_onset_TSP(percentChange_pooledAnimals, 'catches', plot_3sigCI, first_exits_3sigCI, first_exits_y_scatter, baseline_stats, baseline_frames, TGB_bucket_raw)
find_onset_TSP(percentChange_pooledAnimals, 'misses', plot_3sigCI, first_exits_3sigCI, first_exits_y_scatter, baseline_stats, baseline_frames, TGB_bucket_raw)
# make boxplots to show distribution of "onset of tentacle shot pattern"
boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'all', first_exits_3sigCI, first_exits_y_scatter, 3, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)
boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'catches', first_exits_3sigCI, first_exits_y_scatter, 3, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)
boxplots_of_TSP_onset('ProcessCuttlePython', 'PercentChange', 'power at frequency', 'misses', first_exits_3sigCI, first_exits_y_scatter, 3, baseline_frames, TGB_bucket_raw, today_dateTime, plots_folder)
# FIN