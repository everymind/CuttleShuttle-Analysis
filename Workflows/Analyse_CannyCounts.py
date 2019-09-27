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
TGB_files = glob.glob(canny_counts_folder + os.sep + "*.csv")
canny_catch = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
canny_miss = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}

canny_catch_baseline = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
canny_miss_baseline = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}

canny_catch_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}
canny_miss_norm = {"L1-H2013-01": [], "L1-H2013-02": [], "L1-H2013-03": [], "L7-H2013-01": [], "L7-H2013-02": []}

canny_catch_normed_avg = {}
canny_miss_normed_avg = {}

canny_catch_std_error = {}
canny_miss_std_error = {}

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
        canny_catch[TGB_animal].append(TGB_smooth)
    if TGB_type == "miss": 
        canny_miss[TGB_animal].append(TGB_smooth)
    # if TGB_type == "catch":
    #      canny_catch[TGB_animal].append(TGB_moment)
    # if TGB_type == "miss": 
    #     canny_miss[TGB_animal].append(TGB_moment)

all_canny = [canny_catch, canny_miss]

# make average baselined canny count for each animal in catch versus miss conditions
baseline_no_of_buckets = 15
# make baseline for each animal, catch vs miss
for canny_type in all_canny: 
    for key in canny_type: 
        TGB_avg = np.nanmean(canny_type[key], axis=0)
        TGB_baseline = np.nanmean(TGB_avg[0:baseline_no_of_buckets])
        if canny_type == canny_catch:
            canny_catch_baseline[key].append(TGB_baseline)
        if canny_type == canny_miss:
            canny_miss_baseline[key].append(TGB_baseline)
# normalize each trial
    for key in canny_type: 
        if canny_type == canny_catch:
            this_baseline = canny_catch_baseline[key][0]
            for trial in canny_type[key]:
                normed_trial = [(float(x-this_baseline)/this_baseline)*100 for x in trial]
                canny_catch_norm[key].append(normed_trial)
        else:
            this_baseline = canny_miss_baseline[key][0]
            for trial in canny_type[key]:
                normed_trial = [(float(x-this_baseline)/this_baseline)*100 for x in trial]
                canny_miss_norm[key].append(normed_trial)
# find normalized avg for each animal, catch vs miss
    for key in canny_type:
        if canny_type == canny_catch: 
            normed_avg = np.nanmean(canny_catch_norm[key], axis=0)
            canny_catch_normed_avg[key] = normed_avg
        if canny_type == canny_miss: 
            normed_avg = np.nanmean(canny_miss_norm[key], axis=0)
            canny_miss_normed_avg[key] = normed_avg

# plot individual animals
image_type_options = ['.png', '.pdf']
for key in canny_catch: 
    canny_std_catch = np.nanstd(canny_catch_norm[key], axis=0)
    canny_N_catch = len(canny_catch_norm[key])
    canny_std_miss = np.nanstd(canny_miss_norm[key], axis=0)
    canny_N_miss = len(canny_miss_norm[key])
    z_val = 1.96 # Z value for 95% confidence interval
    error_catch = z_val*(canny_std_catch/np.sqrt(canny_N_catch))
    error_miss = z_val*(canny_std_miss/np.sqrt(canny_N_miss))

    canny_catch_std_error[key] = [canny_std_catch, error_catch]
    canny_miss_std_error[key] = [canny_std_miss, error_miss]

    catches_mean = canny_catch_normed_avg[key]
    misses_mean = canny_miss_normed_avg[key]

    figure_name = 'CannyEdgeDetector_' + key + "_PercentChange_" + todays_datetime + '.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = "Average percent change in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Animal: " + key + "\n Number of catches: " + str(canny_N_catch) + ", Number of misses: " + str(canny_N_miss)

    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.ylabel("Percent change in number of edges")
    plot_xticks = np.arange(0, len(canny_catch_normed_avg[key]), step=6)
    plt.xticks(plot_xticks, ['%.1f'%((x*10)/60) for x in plot_xticks])
    plt.xlabel("Seconds")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')

    plt.plot(catches_mean.T, linewidth=2, color=[0.0, 0.0, 1.0, 0.8], label='Catch')
    plt.fill_between(range(len(catches_mean)), catches_mean-error_catch, catches_mean+error_catch, alpha=0.5)

    plt.plot(misses_mean.T, linewidth=2, color=[1.0, 0.0, 0.0, 0.8], label='Miss')
    plt.fill_between(range(len(misses_mean)), misses_mean-error_catch, misses_mean+error_catch, alpha=0.5)

    ymin, ymax = plt.ylim()
    plt.plot((TGB_bucket, TGB_bucket), (ymin, ymax), 'g--', linewidth=1)
    plt.text(TGB_bucket-5, ymax-5, "Tentacles Go Ballistic (TGB)", fontsize='x-small', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.legend(loc='upper left')

    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

### POOL ACROSS ANIMALS ### 

total_N_catch = 0
total_N_miss = 0
for canny_type in all_canny:
    for key in canny_type: 


## FIN