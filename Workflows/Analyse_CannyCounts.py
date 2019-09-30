import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import itertools

### FUNCTIONS ###
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
        TGB_bucket = int(np.ceil((len(TGB_moment)/2)/TGB_window))
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
    return TGB_bucket

def normed_avg_canny_count(prey_type, prey_type_str, baseline_len, baseline_catch, baseline_miss, norm_catch, norm_miss, normed_avg_catch, normed_avg_miss):
    # make baseline for each animal, catch vs miss
    for canny_type in range(len(prey_type)): 
        for animal in prey_type[canny_type]: 
            try:
                TGB_avg = np.nanmean(prey_type[canny_type][animal], axis=0)
                TGB_baseline = np.nanmean(TGB_avg[0:baseline_len])
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

def plot_indiv_animals(prey_type, catches_dict, catches_norm, misses_norm, catches_normed_avg, misses_normed_avg, catches_std_error, misses_std_error, TGB_bucket, plots_dir, todays_dt):
    # plot individual animals
    image_type_options = ['.png', '.pdf']
    for animal in catches_dict: 
        try: 
            canny_std_catch = np.nanstd(catches_norm[animal], axis=0)
            canny_N_catch = len(catches_norm[animal])
            canny_std_miss = np.nanstd(misses_norm[animal], axis=0)
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

def plot_pool_all_animals(prey_type, prey_type_str, catches_norm, misses_norm, catches_normed_avg, misses_normed_avg, TGB_bucket, plots_dir, todays_dt): 
    ### POOL ACROSS ANIMALS ### 
    all_catches = []
    all_misses = []
    for canny_type in range(len(prey_type)):
        for animal in prey_type[canny_type]: 
            this_animal_N = len(prey_type[canny_type][animal])
            if canny_type == 0:
                for trial in catches_norm[animal]:
                    all_catches.append(trial)
            else: 
                for trial in misses_norm[animal]:
                    all_misses.append(trial)
    total_N_catch = len(all_catches)
    total_N_miss = len(all_misses)
    all_catches_mean = np.nanmean(all_catches, axis=0)
    all_catches_std = np.nanstd(all_catches, axis=0)
    all_misses_mean = np.nanmean(all_misses, axis=0)
    all_misses_std = np.nanstd(all_misses, axis=0)

    z_val = 1.96 # Z value for 95% confidence interval
    error_all_catches = z_val*(all_catches_std/np.sqrt(total_N_catch))
    error_all_misses = z_val*(all_misses_std/np.sqrt(total_N_miss))

    figure_name = 'CannyEdgeDetector_'+ prey_type_str + 'Trials_AllAnimals_PercentChange_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Average percent change in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Pooled across all animals \n Prey movement type: " + prey_type_str + "\n Number of catches: " + str(total_N_catch) + ", Number of misses: " + str(total_N_miss)

    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.ylabel("Percent change in number of edges")
    plot_xticks = np.arange(0, len(catches_normed_avg[animal]), step=6)
    plt.xticks(plot_xticks, ['%.1f'%((x*10)/60) for x in plot_xticks])
    plt.xlabel("Seconds")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')

    plt.plot(all_misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
    plt.fill_between(range(len(all_misses_mean)), all_misses_mean-error_all_misses, all_misses_mean+error_all_misses, color=[1.0, 0.0, 0.0, 0.3])

    plt.plot(all_catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
    plt.fill_between(range(len(all_catches_mean)), all_catches_mean-error_all_catches, all_catches_mean+error_all_catches, color=[0.0, 0.0, 1.0, 0.3])

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
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows"
current_working_directory = os.getcwd()
canny_counts_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows\CannyCount_csv_smallCrop_Canny2000-7500"

# set up folders
plots_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\CuttleShuttle-Analysis\Workflows\plots"

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
# all
all_catches_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_misses_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_catches_canny_baseline = {}
all_misses_canny_baseline = {}
all_catches_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_misses_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
all_catches_canny_normed_avg = {}
all_misses_canny_normed_avg = {}
all_catches_canny_std_error = {}
all_misses_canny_std_error = {}
# natural
nat_catches_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_canny_baseline = {}
nat_misses_canny_baseline = {}
nat_catches_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_misses_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
nat_catches_canny_normed_avg = {}
nat_misses_canny_normed_avg = {}
nat_catches_canny_std_error = {}
nat_misses_canny_std_error = {}
# patterned
pat_catches_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_canny_baseline = {}
pat_misses_canny_baseline = {}
pat_catches_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_misses_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
pat_catches_canny_normed_avg = {}
pat_misses_canny_normed_avg = {}
pat_catches_canny_std_error = {}
pat_misses_canny_std_error = {}
# causal
caus_catches_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses_canny = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_canny_baseline = {}
caus_misses_canny_baseline = {}
caus_catches_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_misses_canny_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
caus_catches_canny_normed_avg = {}
caus_misses_canny_normed_avg = {}
caus_catches_canny_std_error = {}
caus_misses_canny_std_error = {}

# collect all canny counts and categorize by animal and type (catch vs miss)
TGB_bucket_all = categorize_by_animal_catchVmiss(TGB_all, all_catches_canny, all_misses_canny)
TGB_bucket_nat = categorize_by_animal_catchVmiss(TGB_natural, nat_catches_canny, nat_misses_canny)
TGB_bucket_pat = categorize_by_animal_catchVmiss(TGB_patterned, pat_catches_canny, pat_misses_canny)
TGB_bucket_caus = categorize_by_animal_catchVmiss(TGB_causal, caus_catches_canny, caus_misses_canny)

all_canny = [all_catches_canny, all_misses_canny]
nat_canny = [nat_catches_canny, nat_misses_canny]
pat_canny = [pat_catches_canny, pat_misses_canny]
caus_canny = [caus_catches_canny, caus_misses_canny]

# make average baselined canny count for each animal in catch versus miss conditions
baseline_buckets = 15
# make baseline for each animal, catch vs miss
normed_avg_canny_count(all_canny, "all", baseline_buckets, all_catches_canny_baseline, all_misses_canny_baseline, all_catches_canny_norm, all_misses_canny_norm, all_catches_canny_normed_avg, all_misses_canny_normed_avg)
normed_avg_canny_count(nat_canny, "natural", baseline_buckets, nat_catches_canny_baseline, nat_misses_canny_baseline, nat_catches_canny_norm, nat_misses_canny_norm, nat_catches_canny_normed_avg, nat_misses_canny_normed_avg)
normed_avg_canny_count(pat_canny, "patterned", baseline_buckets, pat_catches_canny_baseline, pat_misses_canny_baseline, pat_catches_canny_norm, pat_misses_canny_norm, pat_catches_canny_normed_avg, pat_misses_canny_normed_avg)
normed_avg_canny_count(caus_canny, "causal", baseline_buckets, caus_catches_canny_baseline, caus_misses_canny_baseline, caus_catches_canny_norm, caus_misses_canny_norm, caus_catches_canny_normed_avg, caus_misses_canny_normed_avg)

# plot individual animals
# all
plot_indiv_animals("all", all_catches_canny, all_catches_canny_norm, all_misses_canny_norm, all_catches_canny_normed_avg, all_misses_canny_normed_avg, all_catches_canny_std_error, all_misses_canny_std_error, TGB_bucket_all, plots_folder, todays_datetime)
# natural
plot_indiv_animals("natural", nat_catches_canny, nat_catches_canny_norm, nat_misses_canny_norm, nat_catches_canny_normed_avg, nat_misses_canny_normed_avg, nat_catches_canny_std_error, nat_misses_canny_std_error, TGB_bucket_nat, plots_folder, todays_datetime)
# patterned
plot_indiv_animals("patterned", pat_catches_canny, pat_catches_canny_norm, pat_misses_canny_norm, pat_catches_canny_normed_avg, pat_misses_canny_normed_avg, pat_catches_canny_std_error, pat_misses_canny_std_error, TGB_bucket_pat, plots_folder, todays_datetime)
# causal
plot_indiv_animals("causal", caus_catches_canny, caus_catches_canny_norm, caus_misses_canny_norm, caus_catches_canny_normed_avg, caus_misses_canny_normed_avg, caus_catches_canny_std_error, caus_misses_canny_std_error, TGB_bucket_caus, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS ### 
# all
plot_pool_all_animals(all_canny, "all", all_catches_canny_norm, all_misses_canny_norm, all_catches_canny_normed_avg, all_misses_canny_normed_avg, TGB_bucket_all, plots_folder, todays_datetime)
# natural
plot_pool_all_animals(nat_canny, "natural", nat_catches_canny_norm, nat_misses_canny_norm, nat_catches_canny_normed_avg, nat_misses_canny_normed_avg, TGB_bucket_nat, plots_folder, todays_datetime)
# patterned
plot_pool_all_animals(pat_canny, "patterned", pat_catches_canny_norm, pat_misses_canny_norm, pat_catches_canny_normed_avg, pat_misses_canny_normed_avg, TGB_bucket_pat, plots_folder, todays_datetime)
# causal
plot_pool_all_animals(caus_canny, "patterned", caus_catches_canny_norm, caus_misses_canny_norm, caus_catches_canny_normed_avg, caus_misses_canny_normed_avg, TGB_bucket_caus, plots_folder, todays_datetime)

## FIN