import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import itertools
from fitter import Fitter
from scipy.stats import recipinvgauss

### FUNCTIONS ###
def categorize_by_animal(TGB_files, all_animals_dict):
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        # downsample 
        TGB_window = 10 
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
        all_animals_dict[TGB_animal].append(TGB_smooth)
    return TGB_buckets

def categorize_by_animal_catchVmiss(TGB_files, catch_dict, miss_dict):
    # collect all canny counts and categorize by animal and type (catch vs miss)
    for TGB_file in TGB_files: 
        TGB_name = TGB_file.split(os.sep)[-1]
        TGB_animal = TGB_name.split("_")[1]
        TGB_type = TGB_name.split("_")[4]
        TGB_moment = np.genfromtxt(TGB_file, dtype=np.float, delimiter=",")
        # downsample 
        TGB_window = 10 
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
        if TGB_type == "catch":
            catch_dict[TGB_animal].append(TGB_smooth)
        if TGB_type == "miss": 
            miss_dict[TGB_animal].append(TGB_smooth)
    return TGB_buckets

def normed_avg_count(prey_type, prey_type_str, baseline_len, baseline_catch, baseline_miss, norm_catch, norm_miss, normed_avg_catch, normed_avg_miss):
    # make baseline for each animal, catch vs miss
    for canny_type in range(len(prey_type)): 
        for animal in prey_type[canny_type]: 
            try:
                TGB_baseline = np.nanmean(prey_type[canny_type][animal][0:baseline_len])
                # normalize each trial
                all_normed_trials = []
                for trial in prey_type[canny_type][animal]:
                    normed_trial = [(float(x-TGB_baseline)/TGB_baseline)*100 for x in trial]
                    all_normed_trials.append(normed_trial)
                # find normalized avg for each animal
                normed_avg = np.nanmean(all_normed_trials, axis=0)
                if canny_type == 0:
                    baseline_catch[animal] = TGB_baseline
                    norm_catch[animal] = all_normed_trials
                    normed_avg_catch[animal] = normed_avg
                if canny_type == 1:
                    baseline_miss[animal] = TGB_baseline
                    norm_miss[animal] = all_normed_trials
                    normed_avg_miss[animal] = normed_avg
            except Exception:
                if canny_type == 0:
                    print("{a} made no catches during {p} prey movement".format(a=animal,p=prey_type_str))
                if canny_type == 1:
                    print("{a} made no misses during {p} prey movement".format(a=animal,p=prey_type_str))

def plot_hist_all_trials(normalized_dict, canny_type_str, plots_dir, todays_dt):    
    num_bins = 30
    for animal in normalized_dict: 
        figure_name = 'CannyEdgeDetector_' + canny_type_str + 'Trials_' + animal + "_PercentChangeFreqHist_" + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = 'Histogram of changes to number of edges in cuttlefish mantle pattern \n Animal: '+ animal + '\n ' + canny_type_str.split('-')[0] + ' trials, ' + canny_type_str.split('-')[1] + ' prey movements'
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        all_n = []
        all_bins = []
        all_patches = []
        for trial in normalized_dict[animal]:
            n, bins, patches = plt.hist(trial, num_bins, alpha=0.3)
            all_n.append(n)
            all_bins.append(bins)
            all_patches.append(patches)
        plt.xlabel('Percent change in edge count, as detected by Canny Edge Detector')
        plt.ylabel('Number of observations')
        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

###
list_of_TS = all_catches_all_animals_raw
canny_type_str = 'Catch-All'
def plot_log10_hist(list_of_TS, canny_type_str, plots_dir, todays_dt):    
    figure_name = 'Hist_' + canny_type_str + 'Trials_Log10_Raw_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Histogram of log10 transformed number of edges in cuttlefish mantle pattern \n ' + canny_type_str.split('-')[0] + ' trials, ' + canny_type_str.split('-')[1] + ' prey movements'
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    for trial in list_of_TS:
        min_of_trial = min(trial)
        if min_of_trial < 0:
            trial = []
        transformed_trial = np.log10(trial)
        n, bins, patches = plt.hist(transformed_trial, alpha=0.3)
    plt.xlabel('Log10(Percent change in edge count), as detected by Canny Edge Detector')
    plt.ylabel('Number of observations')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_sqrt_hist_all_trials(normalized_dict, canny_type_str, plots_dir, todays_dt):    
    num_bins = 30
    for animal in normalized_dict: 
        figure_name = 'CannyEdgeDetector_' + canny_type_str + 'Trials_' + animal + "_sqrt_PercentChangeFreqHist_" + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = 'Histogram of sqrt transformed changes to number of edges in cuttlefish mantle pattern \n Animal: '+ animal + '\n ' + canny_type_str.split('-')[0] + ' trials, ' + canny_type_str.split('-')[1] + ' prey movements'
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        all_n = []
        all_bins = []
        all_patches = []
        for trial in normalized_dict[animal]:
            min_of_trial = min(trial)
            if min_of_trial < 0:
                trial = trial + abs(min_of_trial)
            transformed_trial = np.sqrt(trial)
            n, bins, patches = plt.hist(transformed_trial, num_bins, alpha=0.3)
            all_n.append(n)
            all_bins.append(bins)
            all_patches.append(patches)
        plt.xlabel('sqrt(Percent change in edge count), as detected by Canny Edge Detector')
        plt.ylabel('Number of observations')
        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def plot_hist_mean_trials(normalized_avg_dict, canny_type_str, plots_dir, todays_dt):
    num_bins = 30
    if canny_type_str.split('-')[0] == 'Catch':
        color = 'blue'
    else:
        color = 'red'
    for animal in normalized_avg_dict: 
        figure_name = 'CannyEdgeDetector_' + canny_type_str + '_Mean_' + animal + "_PercentChangeFreqHist_" + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = 'Histogram of changes to number of edges in cuttlefish mantle pattern \n Animal: '+ animal + '\n Mean of ' + canny_type_str.split('-')[0] + ' trials, ' + canny_type_str.split('-')[1] + ' prey movements'
        plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        n, bins, patches = plt.hist(normalized_avg_dict[animal], num_bins, color=color, alpha=0.6)
        plt.xlabel('Percent change in edge count, as detected by Canny Edge Detector')
        plt.ylabel('Probability')
        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def plot_indiv_animals(prey_type, catches_dict, catches_norm, misses_norm, catches_normed_avg, misses_normed_avg, catches_std_error, misses_std_error, TGB_bucket, plots_dir, todays_dt):
    # plot individual animals
    image_type_options = ['.png', '.pdf']
    for animal in catches_dict: 
        try: 
            canny_std_catch = np.nanstd(catches_norm[animal], axis=0, ddof=1)
            canny_N_catch = len(catches_norm[animal])
            canny_std_miss = np.nanstd(misses_norm[animal], axis=0, ddof=1)
            canny_N_miss = len(misses_norm[animal])
            z_val = 1.96 # Z value for 95% confidence interval
            error_catch = z_val*(canny_std_catch/np.sqrt(canny_N_catch))
            error_miss = z_val*(canny_std_miss/np.sqrt(canny_N_miss))

            catches_std_error[animal] = [canny_std_catch, error_catch]
            misses_std_error[animal] = [canny_std_miss, error_miss]

            catches_mean = catches_normed_avg[animal]
            misses_mean = misses_normed_avg[animal]

            figure_name = 'CannyEdgeDetector_'+ prey_type + 'Trials_' + animal + "_PercentChange_" + todays_dt + '.png'
            figure_path = os.path.join(plots_dir, figure_name)
            figure_title = "Average percent change in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Prey Movement type: " + prey_type + "\n Animal: " + animal + "\n Number of catches: " + str(canny_N_catch) + ", Number of misses: " + str(canny_N_miss)
            plt.figure(figsize=(16,9), dpi=200)
            plt.suptitle(figure_title, fontsize=12, y=0.98)
            plt.ylabel("Percent change in number of edges")
            plot_xticks = np.arange(0, len(catches_normed_avg[animal]), step=6)
            plt.xticks(plot_xticks, ['%.1f'%((x*10)/60) for x in plot_xticks])
            plt.xlabel("Seconds")
            #plt.xlabel("Frame number, original framerate = 60fps")
            plt.grid(b=True, which='major', linestyle='-')

            plt.plot(misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
            plt.fill_between(range(len(misses_mean)), misses_mean-error_miss, misses_mean+error_miss, color=[1.0, 0.0, 0.0, 0.3])

            plt.plot(catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
            plt.fill_between(range(len(catches_mean)), catches_mean-error_catch, catches_mean+error_catch, color=[0.0, 0.0, 1.0, 0.3])

            ymin, ymax = plt.ylim()
            plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
            plt.text(TGB_bucket-5, ymax-5, "Tentacles Go Ballistic (TGB)", fontsize='x-small', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
            plt.legend(loc='upper left')

            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        except Exception:
            print("{a} did not make any catches and/or misses during {p} prey movement".format(a=animal,p=prey_type))

def mult_to_list(input_list, multiplier):
    if not np.isnan(input_list).any(): 
        return [x*multiplier for x in input_list]

def plot_pool_all_animals(prey_type, prey_type_str, catches_norm, misses_norm, catches_normed_avg, misses_normed_avg, catches_std, misses_std, TGB_bucket, plots_dir, todays_dt): 
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
                    this_animal_M = catches_normed_avg[animal]
                    this_animal_std = catches_std[animal][0]
                    all_catches.append((this_animal_M, this_animal_N_TS))
                    catches_std_sq[animal] = (this_animal_std**2, this_animal_N_TS)
                else: 
                    this_animal_M = misses_normed_avg[animal]
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
                    this_deviation = catches_normed_avg[animal] - catches_combined_mean
                    catches_deviations_sq[animal] = this_deviation**2
                else:
                    this_deviation = misses_normed_avg[animal] - misses_combined_mean
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
    figure_title = "Average percent change in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Pooled across all animals \n Prey movement type: " + prey_type_str + "\n Number of catches: " + str(catches_combined_N) + ", Number of misses: " + str(misses_combined_N)

    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.ylabel("Percent change in number of edges")
    plot_xticks = np.arange(0, len(catches_combined_mean), step=6)
    plt.xticks(plot_xticks, ['%.1f'%((x*10)/60) for x in plot_xticks])
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

### BEGIN ANALYSIS ###
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
#root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows"
#canny_counts_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows\CannyCount_csv_smallCrop_Canny2000-7500"
#plots_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows\plots"

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
all_catches_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_misses_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_catches_normed_avg = {}
all_misses_normed_avg = {}
all_catches_std_error = {}
all_misses_std_error = {}
# natural, by catches v misses
nat_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_baseline = {}
nat_misses_baseline = {}
nat_catches_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_normed_avg = {}
nat_misses_normed_avg = {}
nat_catches_std_error = {}
nat_misses_std_error = {}
# patterned, by catches v misses
pat_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_baseline = {}
pat_misses_baseline = {}
pat_catches_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_normed_avg = {}
pat_misses_normed_avg = {}
pat_catches_std_error = {}
pat_misses_std_error = {}
# causal, by catches v misses
caus_catches = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_baseline = {}
caus_misses_baseline = {}
caus_catches_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_normed_avg = {}
caus_misses_normed_avg = {}
caus_catches_std_error = {}
caus_misses_std_error = {}

# collect all canny counts and categorize by animal
TGB_bucket_all = categorize_by_animal(TGB_all, all_TS)

# collect all canny counts and categorize by animal and type (catch vs miss)
TGB_bucket_all_CM = categorize_by_animal_catchVmiss(TGB_all, all_catches, all_misses)
TGB_bucket_nat_CM = categorize_by_animal_catchVmiss(TGB_natural, nat_catches, nat_misses)
TGB_bucket_pat_CM = categorize_by_animal_catchVmiss(TGB_patterned, pat_catches, pat_misses)
TGB_bucket_caus_CM = categorize_by_animal_catchVmiss(TGB_causal, caus_catches, caus_misses)

allTS = [all_catches, all_misses]
nat = [nat_catches, nat_misses]
pat = [pat_catches, pat_misses]
caus = [caus_catches, caus_misses]

all_catches_all_animals_raw = []
all_misses_all_animals_raw = []
for animal in all_catches:
    for trial in all_catches[animal]:
        all_catches_all_animals_raw.append(trial)
for animal in all_misses:
    for trial in all_misses[animal]:
        all_misses_all_animals_raw.append(trial)
all_catches_all_animals_raw = np.array(all_catches_all_animals_raw)
all_misses_all_animals_raw = np.array(all_misses_all_animals_raw)
all_TS_all_animals_raw = np.concatenate([all_catches_all_animals_raw, all_misses_all_animals_raw])

## fit raw data to a statistical distribution
#f_catches = Fitter(all_catches_all_animals_raw)
#f_catches.fit()
#f_catches.summary()
## f_catches.summary()
##               sumsquare_error
## recipinvgauss     5.034673e-12
## halfgennorm       5.264162e-12
## exponpow          6.477840e-12
## gamma             7.005844e-12
## erlang            7.109439e-12
#f_misses = Fitter(all_misses_all_animals_raw)
#f_misses.fit()
#f_misses.summary()
## f_misses.summary()
##               sumsquare_error
## recipinvgauss     3.725484e-12
## exponweib         3.869671e-12
## beta              5.570691e-12
## gengamma          5.616730e-12
## erlang            5.632132e-12
#f_all = Fitter(all_TS_all_animals_raw)
#f_all.fit()
#f_all.summary()
#f_all.summary()
##               sumsquare_error
##halfgennorm       2.824153e-12
##exponpow          2.859841e-12
##chi               3.343348e-12
##recipinvgauss     3.430147e-12
##johnsonsu         3.533115e-12


# plot histograms for each animal to check distribution of dataset
## combined histogram of edge counts from all trials
plot_log10_hist(all_catches_all_animals_raw, 'Catch-All', plots_folder, todays_datetime)

# make average baselined canny count for each animal in catch versus miss conditions
baseline_buckets = 15
# make baseline for each animal, catch vs miss
normed_avg_count(allTS, "all", baseline_buckets, all_catches_baseline, all_misses_baseline, all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg)
normed_avg_count(nat, "natural", baseline_buckets, nat_catches_baseline, nat_misses_baseline, nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg)
normed_avg_count(pat, "patterned", baseline_buckets, pat_catches_baseline, pat_misses_baseline, pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg)
normed_avg_count(caus, "causal", baseline_buckets, caus_catches_baseline, caus_misses_baseline, caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg)

all_norm = [all_catches_norm, all_misses_norm]
all_normed_avg = [all_catches_normed_avg, all_misses_normed_avg]
# shuffle test
### POOL ACROSS ANIMALS and TIME BINS ### 
all_animals_normed_avg_catches = [] 
all_animals_normed_avg_misses = []
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]: 
        this_animal_N_TS = len(all_norm[canny_type][animal])
        if this_animal_N_TS != 0:
            for trial in all_norm[canny_type][animal]:
                for binned_count in trial:
                    if canny_type == 0:
                        all_animals_normed_avg_catches.append(binned_count)
                    else: 
                        all_animals_normed_avg_misses.append(binned_count)
Catch_Group = np.array(all_animals_normed_avg_catches)
Miss_Group = np.array(all_animals_normed_avg_misses)
# Observed performance
OPerf = np.mean(Catch_Group) - np.mean(Miss_Group)
# Shuffle the dataset and compare means again
num_of_shuffles = 10000
SPerf = np.zeros((num_of_shuffles,1))
All_Group = np.concatenate([Catch_Group, Miss_Group])
for shuff in range(num_of_shuffles):
    shuff_response = np.random.permutation(All_Group)
    SPerf[shuff] = np.mean(shuff_response[0:len(Catch_Group)]) - np.mean(shuff_response[len(Catch_Group)+1:len(All_Group)])
# p-value of shuffle test
pVal = np.mean(SPerf**2 >= OPerf**2)
# sigma
sigma_shuff = (OPerf - np.mean(SPerf))/np.std(SPerf)

# plot individual animals
# all
plot_indiv_animals("all", all_catches, all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg, all_catches_std_error, all_misses_std_error, TGB_bucket_all_CM, plots_folder, todays_datetime)
# natural
plot_indiv_animals("natural", nat_catches, nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_nat_CM, plots_folder, todays_datetime)
# patterned
plot_indiv_animals("patterned", pat_catches, pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_pat_CM, plots_folder, todays_datetime)
# causal
plot_indiv_animals("causal", caus_catches, caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_caus_CM, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS ### 
# all
plot_pool_all_animals(allTS, "all", all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg, all_catches_std_error, all_misses_std_error, TGB_bucket_all_CM, plots_folder, todays_datetime)
# natural
plot_pool_all_animals(nat, "natural", nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_nat_CM, plots_folder, todays_datetime)
# patterned
plot_pool_all_animals(pat, "patterned", pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_pat_CM, plots_folder, todays_datetime)
# causal
plot_pool_all_animals(caus, "causal", caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_caus_CM, plots_folder, todays_datetime)

## FIN