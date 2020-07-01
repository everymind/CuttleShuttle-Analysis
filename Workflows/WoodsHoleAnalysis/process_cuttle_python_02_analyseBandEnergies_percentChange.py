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

def plot_percentChange_indiv_animals_allFreq(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_TS_dict, TGB_bucket, baseline_len, plots_dir, todays_dt):
    # plot individual animals
    img_type = ['.png', '.pdf']
    for animal in allA_TS_dict.keys(): 
        try:
            N_TS = len(allA_TS_dict[animal][0]['trials'])
            # set fig path and title
            if len(prey_type_str.split(' '))>1:
                figure_name = analysis_type_str+'_'+preprocess_str+'_'+animal+'_allFreqBand'+'_'+prey_type_str.split(' ')[1]+'Trials_'+todays_dt+img_type[0]
            else:
                figure_name = analysis_type_str+'_'+preprocess_str+'_'+animal+'_allFreqBand'+'_'+prey_type_str+'Trials_'+todays_dt+img_type[0]
            figure_path = os.path.join(plots_dir, figure_name)
            figure_title = 'Mean percent change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Individual trials plotted with more transparent traces \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, Animal: {a}\n Number of tentacle shots: {Nts}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, a=animal, Nts=str(N_TS))
            # setup fig
            plt.figure(figsize=(16,9), dpi=200)
            plt.suptitle(figure_title, fontsize=12, y=0.99)
            plt.ylabel("Percent change from baseline in power")
            plot_xticks = np.arange(0, len(allA_TS_dict[animal][0]['trials'][0]), step=60)
            plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
            #plt.xlim(0,180)
            plt.ylim(-200, 200)
            plt.xlabel("Seconds")
            plt.grid(b=True, which='major', linestyle='-')
            N_freq_bands = len(allA_TS_dict[animal].keys())
            colors = pl.cm.jet(np.linspace(0,1,N_freq_bands))
            for freq_band in allA_TS_dict[animal].keys():
                TS_mean_frame = allA_TS_dict[animal][freq_band]['mean frame']
                # plot percent change in power at frequency band
                for trial in allA_TS_dict[animal][freq_band]['trials']:
                    plt.plot(trial, linewidth=1, color=colors[freq_band], alpha=0.03)
                plt.plot(TS_mean_frame.T, linewidth=2, color=colors[freq_band], alpha=0.5, label='Freq Band {fb}'.format(fb=freq_band))
            # plot events
            ymin, ymax = plt.ylim()
            plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
            plt.text(baseline_len, ymax-100, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
            plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
            plt.text(TGB_bucket, ymax-50, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
            plt.legend(loc='upper left')
            # save fig
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        except Exception:
            plt.close()
            print("{a} did not make any catches and/or misses during {p} prey movement".format(a=animal,p=prey_type_str))

analysis_type_str = 'ProcessCuttlePython'
preprocess_str = 'PercentChange_Frame'
metric_str = 'power at frequency band'
prey_type_str = 'all'
allA_mean_var_dict = percentChange_allAnimals
TGB_bucket = TGB_bucket_raw
baseline_len = baseline_frames
plots_dir = plots_folder
todays_dt = today_dateTime

def plot_percentChange_pooled_animals_allFreq(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_mean_var_dict, TGB_bucket, baseline_len, plots_dir, todays_dt):
    img_type = ['.png', '.pdf']
    # calculate total number of tentacle shots
    N_TS = 0
    for animal in allA_mean_var_dict['N'][0]:
        N_TS += animal
    # calculate mean and variance across all animals
    pooled_means = {}
    pooled_vars = {}
    for freq_band in allA_mean_var_dict['N'].keys():
        # find pooled mean
        pooled_mean_numerator = []
        pooled_denominator = []
        for animal in range(len(allA_mean_var_dict['N'][freq_band])):
            this_animal_mean_numerator = allA_mean_var_dict['N'][freq_band][animal]*allA_mean_var_dict['Mean'][freq_band][animal]
            pooled_mean_numerator.append(this_animal_mean_numerator)
            pooled_denominator.append(allA_mean_var_dict['N'][freq_band][animal])
        this_freq_pooled_mean = np.sum(pooled_mean_numerator, axis=0)/np.sum(pooled_denominator)
        # find pooled variance
        pooled_var_numerator = []
        for animal in range(len(allA_mean_var_dict['N'][freq_band])):
            this_animal_var_numerator = []
            for trial in allA_mean_var_dict['trials'][freq_band][animal]:
                this_trial_var = np.square(trial-this_freq_pooled_mean)
                this_animal_var_numerator.append(this_trial_var)
            pooled_var_numerator.append(np.sum(this_animal_var_numerator, axis=0))
        this_freq_pooled_var = np.sum(pooled_var_numerator, axis=0)/(np.sum(pooled_denominator)-1)
        pooled_means[freq_band] = this_freq_pooled_mean
        pooled_vars[freq_band] = this_freq_pooled_var
    # set fig path and title
    if len(prey_type_str.split(' '))>1:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_allFreqBand_'+prey_type_str.split(' ')[1]+'Trials_'+todays_dt+img_type[0]
    else:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_allFreqBand_'+prey_type_str+'Trials_'+todays_dt+img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Mean percent change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Individual trials plotted with more transparent traces \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, Nts=str(N_TS))
    # setup fig
    plt.figure(figsize=(16,16), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.ylabel("Percent change from baseline in power")
    plot_xticks = np.arange(0, len(allA_mean_var_dict['Mean'][0][0]), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    #plt.xlim(0,180)
    plt.ylim(-600, 600)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    N_freq_bands = len(allA_mean_var_dict['N'].keys())
    colors = pl.cm.jet(np.linspace(0,1,N_freq_bands))
    for freq_band in pooled_means.keys():
        x_frames = range(360)
        upper_var = pooled_means[freq_band] + pooled_vars[freq_band]
        lower_var = pooled_means[freq_band] - pooled_vars[freq_band]
        plt.fill_between(x_frames, upper_var, lower_var, color=colors[freq_band], alpha=0.03)
        plt.plot(pooled_means[freq_band], linewidth=2, color=colors[freq_band], alpha=0.5, label='Freq Band {fb}'.format(fb=freq_band))
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-100, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-50, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
    # save fig
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_percentChange_pooled_animals_someFreq(analysis_type_str, preprocess_str, metric_str, prey_type_str, allA_mean_var_dict, list_of_freqs_to_plot, TGB_bucket, baseline_len, plots_dir, todays_dt):
    img_type = ['.png', '.pdf']
    # calculate total number of tentacle shots
    N_TS = 0
    for animal in allA_mean_var_dict['N'][0]:
        N_TS += animal
    # calculate mean and variance across all animals
    pooled_means = {}
    pooled_vars = {}
    for freq_band in allA_mean_var_dict['N'].keys():
        # find pooled mean
        pooled_mean_numerator = []
        pooled_denominator = []
        for animal in range(len(allA_mean_var_dict['N'][freq_band])):
            this_animal_mean_numerator = allA_mean_var_dict['N'][freq_band][animal]*allA_mean_var_dict['Mean'][freq_band][animal]
            pooled_mean_numerator.append(this_animal_mean_numerator)
            pooled_denominator.append(allA_mean_var_dict['N'][freq_band][animal])
        this_freq_pooled_mean = np.sum(pooled_mean_numerator, axis=0)/np.sum(pooled_denominator)
        # find pooled variance
        pooled_var_numerator = []
        for animal in range(len(allA_mean_var_dict['N'][freq_band])):
            this_animal_var_numerator = []
            for trial in allA_mean_var_dict['trials'][freq_band][animal]:
                this_trial_var = np.square(trial-this_freq_pooled_mean)
                this_animal_var_numerator.append(this_trial_var)
            pooled_var_numerator.append(np.sum(this_animal_var_numerator, axis=0))
        this_freq_pooled_var = np.sum(pooled_var_numerator, axis=0)/(np.sum(pooled_denominator)-1)
        pooled_means[freq_band] = this_freq_pooled_mean
        pooled_vars[freq_band] = this_freq_pooled_var
    # set fig path and title
    freq_bands_str = 'freqBands'
    for freq_band in list_of_freqs_to_plot:
        freq_bands_str += str(freq_band)+'-'
    if len(prey_type_str.split(' '))>1:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_'+freq_bands_str+'_'+prey_type_str.split(' ')[1]+'Trials_'+todays_dt+img_type[0]
    else:
        figure_name = analysis_type_str+'_'+preprocess_str+'_allAnimals_'+freq_bands_str+'_'+prey_type_str+'Trials_'+todays_dt+img_type[0]
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Mean percent change from baseline of {m} in ROI on cuttlefish mantle during tentacle shots, as detected by {at}\n Individual trials plotted with more transparent traces \n Baseline: mean of {m} from t=0 to t={b} seconds \n Prey Movement type: {p}, pooled across all animals\n Number of tentacle shots: {Nts}, showing frequency bands {fb}'.format(m=metric_str, at=analysis_type_str, b=str(baseline_len/60), p=prey_type_str, Nts=str(N_TS), fb=freq_bands_str[9:])
    # setup fig
    plt.figure(figsize=(16,16), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.99)
    plt.ylabel("Percent change from baseline in power")
    plot_xticks = np.arange(0, len(allA_mean_var_dict['Mean'][0][0]), step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    #plt.xlim(0,180)
    plt.ylim(-600, 600)
    plt.xlabel("Seconds")
    plt.grid(b=True, which='major', linestyle='-')
    N_freq_bands = len(allA_mean_var_dict['N'].keys())
    colors = pl.cm.jet(np.linspace(0,1,N_freq_bands))
    for freq_band in list_of_freqs_to_plot:
        x_frames = range(360)
        upper_var = pooled_means[freq_band] + pooled_vars[freq_band]
        lower_var = pooled_means[freq_band] - pooled_vars[freq_band]
        plt.fill_between(x_frames, upper_var, lower_var, color=colors[freq_band], alpha=0.05)
        plt.plot(pooled_means[freq_band], linewidth=2, color=colors[freq_band], alpha=0.5, label='Freq Band {fb}'.format(fb=freq_band))
    # plot events
    ymin, ymax = plt.ylim()
    plt.plot((baseline_len, baseline_len), (ymin, ymax), 'm--', linewidth=1)
    plt.text(baseline_len, ymax-100, "End of \nbaseline period", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.35'))
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket, ymax-50, "Tentacles Go Ballistic\n(TGB)", fontsize='small', ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')
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
# frame for moment tentacles go ballistic
TGB_bucket_raw = 180
########################################################
### ------ DATA NORMALIZATION/STANDARDIZATION ------ ###
########################################################
baseline_frames = 150 #frames
# convert power at frequency into percent change from baseline
dailyTS_percentChange = {}
for session_date in all_TS_daily:
    dailyTS_percentChange[session_date] = percent_change_from_baseline(all_TS_daily[session_date], 'all', baseline_frames)
allTS_percentChange = percent_change_from_baseline(all_TS, 'all', baseline_frames)
#######################################################
### ------ PLOT PERCENT CHANGE FROM BASELINE ------ ###
#######################################################
plot_percentChange_indiv_animals_allFreq('ProcessCuttlePython', 'PercentChange_Frame', 'power at frequency band', 'all', allTS_percentChange, TGB_bucket_raw, baseline_frames, plots_folder, today_dateTime)
# pool across all animals to plot mean percent change in each frequency for all animals
percentChange_allAnimals = {'N': {}, 'Mean': {}, 'trials': {}}
for animal in allTS_percentChange.keys():
    for freq_band in allTS_percentChange[animal].keys():
        this_animal_this_freq_N = len(allTS_percentChange[animal][freq_band]['trials'])
        this_animal_this_freq_mean = allTS_percentChange[animal][freq_band]['mean frame']
        this_animal_this_freq_var = allTS_percentChange[animal][freq_band]['std frame']
        percentChange_allAnimals['N'].setdefault(freq_band,[]).append(this_animal_this_freq_N)
        percentChange_allAnimals['Mean'].setdefault(freq_band,[]).append(this_animal_this_freq_mean)
        percentChange_allAnimals['trials'].setdefault(freq_band,[]).append(allTS_percentChange[animal][freq_band]['trials'])
# plot
plot_percentChange_pooled_animals_allFreq('ProcessCuttlePython', 'PercentChange_Frame', 'power at frequency band', 'all', percentChange_allAnimals, TGB_bucket_raw, baseline_frames, plots_folder, today_dateTime)
# pick out certain frequencies to plot
plot_percentChange_pooled_animals_someFreq('ProcessCuttlePython', 'PercentChange_Frame', 'power at frequency band', 'all', percentChange_allAnimals, [0,1,2], TGB_bucket_raw, baseline_frames, plots_folder, today_dateTime)
plot_percentChange_pooled_animals_someFreq('ProcessCuttlePython', 'PercentChange_Frame', 'power at frequency band', 'all', percentChange_allAnimals, [1,2], TGB_bucket_raw, baseline_frames, plots_folder, today_dateTime)
