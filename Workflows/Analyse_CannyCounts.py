import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import itertools
import scipy
#from fitter import Fitter
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

def normed_avg_count(prey_type, prey_type_str, baseline_len, baseline_catch, baseline_miss, norm_catch, norm_miss, normed_avg_catch, normed_avg_miss):
    # make baseline for each animal, catch vs miss
    for canny_type in range(len(prey_type)): 
        for animal in prey_type[canny_type]: 
            try:
                # normalize each trial
                all_normed_trials = []
                for trial in prey_type[canny_type][animal]:
                    TGB_baseline = np.nanmean(trial[0:baseline_len])
                    normed_trial = [float(x-TGB_baseline) for x in trial]
                    all_normed_trials.append(normed_trial)
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
    for animal in catches_dict.keys(): 
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
            figure_title = "Average change from baseline in number of edges (with 95% CI) in cuttlefish mantle pattern during tentacle shots, as detected by Canny Edge Detector \n Baseline: mean of edge counts from t=0 to t=2 seconds \n Prey Movement type: " + prey_type + ", Animal: " + animal + "\n Number of catches: " + str(canny_N_catch) + ", Number of misses: " + str(canny_N_miss)
            plt.figure(figsize=(16,9), dpi=200)
            plt.suptitle(figure_title, fontsize=12, y=0.98)
            plt.ylabel("Change from baseline in number of edges")
            plot_xticks = np.arange(0, len(catches_normed_avg[animal]), step=60)
            plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
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
            plt.close()
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

def shuffle_test(Group1, Group2, N_Shuffles, Group1_str, Group2_str, Group1_N, Group2_N, plots_dir, todays_dt):
    # Observed performance
    OPerf = np.mean(Group1) - np.mean(Group2)
    # Shuffle the dataset and compare means again
    num_of_shuffles = N_Shuffles
    SPerf = np.zeros((num_of_shuffles,1))
    All_Group = np.concatenate([Group1, Group2])
    for shuff in range(num_of_shuffles):
        shuff_response = np.random.permutation(All_Group)
        SPerf[shuff] = np.mean(shuff_response[0:len(Group1)]) - np.mean(shuff_response[len(Group1)+1:len(All_Group)])
    # p-value of shuffle test
    pVal = np.mean(SPerf**2 >= OPerf**2)
    # sigma
    sigma_shuff = (OPerf - np.mean(SPerf))/np.std(SPerf)
    # show histogram of diffs of shuffled means
    figure_name = 'ShuffleTest_'+ Group1_str + '_' + Group2_str + '_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = "Histogram of the differences in means of randomly labeled data, Number of shuffles = " + str(N_Shuffles) + "\n Group 1: Normalized edge counts, " + Group1_str + ", N = " + str(Group1_N) + "\n Group 2: Normalized edge counts, " + Group2_str + ", N = " + str(Group2_N) + "\n P-value of shuffle test: " + str(pVal) + ", Sigma of shuffle test: " + str(sigma_shuff)
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.hist(SPerf)
    ymin, ymax = plt.ylim()
    plt.plot((OPerf, OPerf), (ymin, ymax), 'g--', linewidth=1)
    plt.text(OPerf, ymax-5, "Difference of Labeled Means = " + str(OPerf), fontsize='x-small', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.35'))
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return pVal, sigma_shuff

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
categorize_by_animal(TGB_all, all_TS)
# collect all canny counts and categorize by animal and type (catch vs miss)
categorize_by_animal_catchVmiss(TGB_all, all_catches, all_misses)
categorize_by_animal_catchVmiss(TGB_natural, nat_catches, nat_misses)
categorize_by_animal_catchVmiss(TGB_patterned, pat_catches, pat_misses)
categorize_by_animal_catchVmiss(TGB_causal, caus_catches, caus_misses)

all_raw = [all_catches, all_misses]
nat_raw= [nat_catches, nat_misses]
pat_raw= [pat_catches, pat_misses]
caus_raw= [caus_catches, caus_misses]

# make baselined canny count for each animal in catch versus miss conditions
baseline_buckets = 120
# make baseline for each animal, catch vs miss
normed_avg_count(all_raw, "all", baseline_buckets, all_catches_baseline, all_misses_baseline, all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg)
normed_avg_count(nat_raw, "natural", baseline_buckets, nat_catches_baseline, nat_misses_baseline, nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg)
normed_avg_count(pat_raw "patterned", baseline_buckets, pat_catches_baseline, pat_misses_baseline, pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg)
normed_avg_count(caus_raw "causal", baseline_buckets, caus_catches_baseline, caus_misses_baseline, caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg)

all_norm = [all_catches_norm, all_misses_norm]
nat_norm = [nat_catches_norm, nat_misses_norm]
pat_norm = [pat_catches_norm, pat_misses_norm]
caus_norm = [caus_catches_norm, caus_misses_norm]

"""# plot all traces
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        for trial in all_norm[canny_type][animal]:
            if canny_type == 0:
                plt.plot(trial, color='blue', alpha=0.1)
            else:
                plt.plot(trial, color='red', alpha=0.1)
plt.show()"""

"""# pool raw data to find distribution of data
all_catches_all_animals_raw = []
all_misses_all_animals_raw = []
for animal in all_catches:
    for trial in all_catches[animal]:
        all_catches_all_animals_raw.append(trial)
for animal in all_misses:
    for trial in all_misses[animal]:
        all_misses_all_animals_raw.append(trial)
all_catches_all_animals_raw = np.array(all_catches_all_animals_raw)
all_catches_raw_zeros_excluded = all_catches_all_animals_raw[all_catches_all_animals_raw != 0]
all_misses_all_animals_raw = np.array(all_misses_all_animals_raw)
all_misses_raw_zeros_excluded = all_misses_all_animals_raw[all_misses_all_animals_raw != 0]
all_TS_all_animals_raw = np.concatenate([all_catches_all_animals_raw, all_misses_all_animals_raw])
all_TS_raw_zeros_excluded = all_TS_all_animals_raw[all_TS_all_animals_raw != 0]

## fit raw data to a statistical distribution
f_catches = Fitter(all_catches_all_animals_raw)
f_catches.fit()
f_catches.summary()
## f_catches.summary()
##           sumsquare_error
##gengamma      7.793988e-12
##levy          7.988078e-12
##exponpow      8.425763e-12
##gamma         1.065244e-11
##exponweib     1.070720e-11
f_catches_zeros_excluded = Fitter(all_catches_raw_zeros_excluded)
f_catches_zeros_excluded.fit()
f_catches_zeros_excluded.summary()
## f_catches_zeros_excluded.summary()
##            sumsquare_error
##johnsonsb      7.126538e-13
##chi            1.051106e-12
##gausshyper     1.214453e-12
##chi2           1.302191e-12
##erlang         1.315014e-12
f_misses = Fitter(all_misses_all_animals_raw)
f_misses.fit()
f_misses.summary()
## f_misses.summary()
##          sumsquare_error
##expon        2.741688e-11
##gumbel_r     3.666497e-11
##cauchy       3.876534e-11
##norm         3.905432e-11
##gumbel_l     4.161271e-11
f_misses_zeros_excluded = Fitter(all_misses_raw_zeros_excluded)
f_misses_zeros_excluded.fit()
f_misses_zeros_excluded.summary()
##f_misses_zeros_excluded.summary()
##              sumsquare_error
##expon            1.973342e-12
##halfcauchy       2.779816e-12
##halflogistic     2.831054e-12
##halfnorm         3.153319e-12
##levy             4.085157e-12
f_all = Fitter(all_TS_all_animals_raw)
f_all.fit()
f_all.summary()
##f_all.summary()
##          sumsquare_error
##erlang       6.205514e-12
##gengamma     7.076662e-12
##beta         8.769084e-12
##levy         9.046809e-12
##chi          9.261221e-12
f_all_zeros_excluded = Fitter(all_TS_raw_zeros_excluded)
f_all_zeros_excluded.fit()
f_all_zeros_excluded.summary()
##f_all_zeros_excluded.summary()
##           sumsquare_error
##expon         2.046319e-12
##halfnorm      3.919703e-12
##moyal         5.549140e-12
##kstwobign     5.999799e-12
##gumbel_r      6.054823e-12 """

## visualize the data
TGB_bucket_raw = 180
# all
plot_indiv_animals("all", all_catches, all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg, all_catches_std_error, all_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# natural
plot_indiv_animals("natural", nat_catches, nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# patterned
plot_indiv_animals("patterned", pat_catches, pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# causal
plot_indiv_animals("causal", caus_catches, caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)

### POOL ACROSS ANIMALS ### 
# all
plot_pool_all_animals(all_raw, "all", all_catches_norm, all_misses_norm, all_catches_normed_avg, all_misses_normed_avg, all_catches_std_error, all_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# natural
plot_pool_all_animals(nat_raw "natural", nat_catches_norm, nat_misses_norm, nat_catches_normed_avg, nat_misses_normed_avg, nat_catches_std_error, nat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# patterned
plot_pool_all_animals(pat_raw "patterned", pat_catches_norm, pat_misses_norm, pat_catches_normed_avg, pat_misses_normed_avg, pat_catches_std_error, pat_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)
# causal
plot_pool_all_animals(caus_raw "causal", caus_catches_norm, caus_misses_norm, caus_catches_normed_avg, caus_misses_normed_avg, caus_catches_std_error, caus_misses_std_error, TGB_bucket_raw, plots_folder, todays_datetime)

# Z score the data
# first: z-score for each animal, across all timebins and trials
allTS_normed_perAnimal = {}
zScore_perAnimal = {}
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]:
        for trial in all_norm[canny_type][animal]:
            allTS_normed_perAnimal.setdefault(animal,[]).append(trial)
for animal in allTS_normed_perAnimal:
    zScore_perAnimal[animal] = stats.zscore(allTS_normed_perAnimal[animal], axis=1, ddof=1)
## visualize
for animal in zScore_perAnimal:
    for trial in zScore_perAnimal[animal]:
        plt.plot(trial, alpha=0.2)
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
    zScore_perA_perTB[animal] = {}
    for timebin in allTS_normed_perA_perTB[animal]:
        zScore_perA_perTB[animal][timebin] = stats.zscore(allTS_normed_perA_perTB[animal][timebin], ddof=1)
## visualize
for animal in zScore_perA_perTB:
    for timebin in zScore_perA_perTB[animal]:
        x_pos = np.array([timebin]*len(zScore_perA_perTB[animal][timebin]))
        plt.scatter(x_pos, zScore_perA_perTB[animal][timebin], alpha=0.2)
    plt.show()


# shuffle test
### POOL ACROSS ANIMALS and TIME BINS
all_animals_normed_catches = []
all_animals_normed_catches_N = 0
all_animals_normed_misses = []
all_animals_normed_misses_N = 0
### POOL ACROSS ALL ANIMALS, before and after TGB
all_animals_normed_preTGB_catches = []
all_animals_normed_preTGB_catches_N = 0
all_animals_normed_preTGB_misses = []
all_animals_normed_preTGB_misses_N = 0
all_animals_normed_postTGB_catches = []
all_animals_normed_postTGB_catches_N = 0
all_animals_normed_postTGB_misses = []
all_animals_normed_postTGB_misses_N = 0
#
for canny_type in range(len(all_norm)):
    for animal in all_norm[canny_type]: 
        this_animal_N_TS = len(all_norm[canny_type][animal])
        if canny_type == 0:
            all_animals_normed_catches_N = all_animals_normed_catches_N + this_animal_N_TS
            all_animals_normed_preTGB_catches_N = all_animals_normed_preTGB_catches_N + this_animal_N_TS
            all_animals_normed_postTGB_catches_N = all_animals_normed_postTGB_catches_N + this_animal_N_TS
        else: 
            all_animals_normed_misses_N = all_animals_normed_misses_N + this_animal_N_TS
            all_animals_normed_preTGB_misses_N = all_animals_normed_preTGB_misses_N + this_animal_N_TS
            all_animals_normed_postTGB_misses_N = all_animals_normed_postTGB_misses_N + this_animal_N_TS
        if this_animal_N_TS != 0:
            for trial in all_norm[canny_type][animal]:
                ### POOL ACROSS ANIMALS and TIME BINS
                for binned_count in trial:
                    if canny_type == 0:
                        all_animals_normed_catches.append(binned_count)
                    else: 
                        all_animals_normed_misses.append(binned_count)
                ### POOL ACROSS ALL ANIMALS, before and after TGB
                TGB_moment = int(len(trial)/2)
                for binned_count in trial[0:TGB_moment-1]:
                    if canny_type == 0:
                        all_animals_normed_preTGB_catches.append(binned_count)
                    else:
                        all_animals_normed_preTGB_misses.append(binned_count)
                for binned_count in trial[TGB_moment:-1]:
                    if canny_type == 0:
                        all_animals_normed_postTGB_catches.append(binned_count)
                    else:
                        all_animals_normed_postTGB_misses.append(binned_count)
# all TS = 10
All_Catch_Group = np.array(all_animals_normed_catches)
All_Miss_Group = np.array(all_animals_normed_misses)
pVal_all_animals_allTS, sigma_all_animals_allTS = shuffle_test(All_Catch_Group, All_Miss_Group, 20000, "AllCatch", "AllMiss", all_animals_normed_catches_N, all_animals_normed_misses_N, plots_folder, todays_datetime)
# all preTGB = 4
All_Catch_PreTGB = np.array(all_animals_normed_preTGB_catches)
All_Miss_PreTGB = np.array(all_animals_normed_preTGB_misses)
pVal_all_animals_preTGB, sigma_all_animals_preTGB = shuffle_test(All_Catch_PreTGB, All_Miss_PreTGB, 20000, "AllCatch-preTGB", "AllMiss-preTGB", all_animals_normed_preTGB_catches_N, all_animals_normed_preTGB_misses_N, plots_folder, todays_datetime)
# all postTGB = 4
All_Catch_PostTGB = np.array(all_animals_normed_postTGB_catches)
All_Miss_PostTGB = np.array(all_animals_normed_postTGB_misses)
pVal_all_animals_postTGB, sigma_all_animals_postTGB = shuffle_test(All_Catch_PostTGB, All_Miss_PostTGB, 20000, "AllCatch-postTGB", "AllMiss-postTGB", all_animals_normed_postTGB_catches_N, all_animals_normed_postTGB_misses_N,  plots_folder, todays_datetime)






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