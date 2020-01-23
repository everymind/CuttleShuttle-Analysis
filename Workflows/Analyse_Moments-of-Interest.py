import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math
import sys
import itertools
import scipy
import scipy.signal
from scipy import stats

### FUNCTIONS ###
def convert_timestamps_to_secs_from_start(allA_timestamps_dict, list_of_MOIs):
    converted_dict = {}
    for animal in allA_timestamps_dict:
        converted_dict[animal] = {}
        all_session_dates = []
        all_session_lens = []
        all_food_offerings = []
        all_homebases = []
        all_orientations = []
        all_tentacle_shots = []
        all_catches = []
        all_mois = [all_food_offerings, all_homebases, all_orientations, all_tentacle_shots, all_catches]
        for session_date in sorted(allA_timestamps_dict[animal].keys()):
            all_session_dates.append(session_date)
            start_ts = allA_timestamps_dict[animal][session_date]['session vids'][0]
            end_ts = allA_timestamps_dict[animal][session_date]['session vids'][-1]
            session_len = (end_ts - start_ts).total_seconds()
            all_session_lens.append(session_len)
            if len(allA_timestamps_dict[animal][session_date].keys())==1 and 'session vids' in allA_timestamps_dict[animal][session_date]:
                print("No moments of interest for animal {a} on {s}".format(a=animal,s=session_date))
                for moi in range(len(list_of_MOIs)):
                    all_mois[moi].append([])
            else:
                for moi in range(len(list_of_MOIs)):
                    if list_of_MOIs[moi] in allA_timestamps_dict[animal][session_date]:
                        this_session_mois = []
                        if allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]].size <= 1:
                            timestamp = allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]]
                            time_diff = (timestamp - start_ts).total_seconds()
                            this_session_mois.append(time_diff)
                        else:
                            for timestamp in allA_timestamps_dict[animal][session_date][list_of_MOIs[moi]]:
                                time_diff = (timestamp - start_ts).total_seconds()
                                this_session_mois.append(time_diff)
                        all_mois[moi].append(this_session_mois)
                    else:
                        all_mois[moi].append([])
        converted_dict[animal]['session dates'] = all_session_dates
        converted_dict[animal]['session durations'] = all_session_lens
        for moi in range(len(list_of_MOIs)):
            converted_dict[animal][list_of_MOIs[moi]] = all_mois[moi]
    return converted_dict

def plot_timeline_MOIs(dict_allAnimals_allMOIs, list_of_MOIs, animal_names_dict, plots_dir, todays_dt):
    for animal in dict_allAnimals_allMOIs:
        all_session_lens = dict_allAnimals_allMOIs[animal]['session durations']
        all_session_dates = dict_allAnimals_allMOIs[animal]['session dates']
        # prepare for possible MOIs
        all_homebases = []
        all_orientations = []
        all_tentacle_shots = []
        all_food_offerings = []
        all_catches = []
        for moi in list_of_MOIs:
            if moi=='homebase':
                all_homebases = dict_allAnimals_allMOIs[animal]['homebase']
            if moi=='orients':
                all_orientations = dict_allAnimals_allMOIs[animal]['orients']
            if moi=='tentacle shots':
                all_tentacle_shots = dict_allAnimals_allMOIs[animal]['tentacle shots']
            if moi=='food offerings':
                all_food_offerings = dict_allAnimals_allMOIs[animal]['food offerings']
            if moi=='catches':
                all_catches = dict_allAnimals_allMOIs[animal]['catches']
        # remove habituation session
        all_session_dates = all_session_dates[1:]
        all_homebases = all_homebases[1:]
        all_orientations = all_orientations[1:]
        all_tentacle_shots = all_tentacle_shots[1:]
        all_food_offerings = all_food_offerings[1:]
        # reverse lists of MOI times so that first session appears at the top of the plot
        all_session_dates_reversed = all_session_dates[::-1]
        all_homebases_reversed = all_homebases[::-1]
        all_orientations_reversed = all_orientations[::-1]
        all_tentacle_shots_reversed = all_tentacle_shots[::-1]
        all_food_offerings_reversed = all_food_offerings[::-1]
        all_catches_reversed = all_catches[::-1]
        session_len_mins = 37
        plotting_session_len = session_len_mins*60
        # find number of each MOI
        hb_count = sum([len(session) for session in all_homebases_reversed])
        orients_count = sum([len(session) for session in all_orientations_reversed])
        ts_count = sum([len(session) for session in all_tentacle_shots_reversed])
        c_count = sum([len(session) for session in all_catches_reversed])
        fo_count = sum([len(session) for session in all_food_offerings_reversed])
        # set colors
        food_offerings_color = [1.0, 0.0, 1.0, 0.5]
        homebase_color = [1.0, 0.0, 0.0, 0.5]
        orientations_color = [1.0, 0.647, 0.0, 0.6]
        tentacle_shots_color = [0.0, 1.0, 0.0, 0.5]
        catches_color = [0.0, 0.0, 1.0, 0.4]
        # set figure save path and title
        figure_name = 'MomentsOfInterest_'+ animal + '_' + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = 'Moments of interest during hunting session of ' + animal + ', aka ' + animal_names_dict[animal] + '\nReturns to home base: ' + str(hb_count) + '\nNumber of Orientations: ' + str(orients_count) + '\n Number of Tentacle Shots: ' + str(ts_count) + ', Number of Catches: ' + str(c_count)
        # set axes and other figure properties
        ax = plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        plt.xlim(-60,plotting_session_len+60)
        plt.ylim(-1, len(all_session_lens))
        plot_yticks = np.arange(0, len(all_session_lens), 1)
        ytick_labels = ['Day '+str(x+1) for x in range(len(all_session_dates_reversed))]
        plt.yticks(plot_yticks, ytick_labels[::-1])
        plt.ylabel("Session Number")
        plot_xticks = np.arange(0, plotting_session_len, step=60)
        plt.xticks(plot_xticks, ['%d'%(x/60) for x in plot_xticks])
        plt.xlabel("Minute from start of hunting session")
        #plt.xlabel("Frame number, original framerate = 60fps")
        plt.grid(b=True, which='major', linestyle='-')
        # draw moments of interest
        for session in range(len(all_food_offerings_reversed)):
            if len(all_food_offerings_reversed[session]) > 0:
                for time_diff in all_food_offerings_reversed[session]:
                    plt.plot(time_diff, session, Marker='|', markersize=7, color=food_offerings_color)
        for session in range(len(all_homebases_reversed)):
            if len(all_homebases_reversed[session]) > 0:
                for time_diff in all_homebases_reversed[session]:
                    plt.plot(time_diff, session, Marker='d', markersize=7, color=homebase_color)
        for session in range(len(all_orientations_reversed)):
            if len(all_orientations_reversed[session]) > 0:
                for time_diff in all_orientations_reversed[session]:
                    plt.plot(time_diff, session, Marker='^', markersize=7, color=orientations_color)
        for session in range(len(all_tentacle_shots_reversed)):
            if len(all_tentacle_shots_reversed[session]) > 0:
                for time_diff in all_tentacle_shots_reversed[session]:
                    plt.plot(time_diff, session, Marker='*', markersize=7, color=tentacle_shots_color)
        for session in range(len(all_catches_reversed)):
            if len(all_catches_reversed[session]) > 0:
                for time_diff in all_catches_reversed[session]:
                    plt.plot(time_diff, session, Marker='*', markersize=7, color=catches_color)
        # create custom legend
        legend_elements = [Line2D([0], [0], marker='|', color=food_offerings_color, label='Food offerings', markerfacecolor=food_offerings_color, markersize=10),
                        Line2D([0], [0], marker='d', color=homebase_color, label='Returns to homebase', markerfacecolor=homebase_color, markersize=10),
                        Line2D([0], [0], marker='^', color=orientations_color, label='Orientations', markerfacecolor=orientations_color, markersize=10),
                        Line2D([0], [0], marker='*', color=tentacle_shots_color, label='Tentacle Shots', markerfacecolor=tentacle_shots_color, markersize=10),
                        Line2D([0], [0], marker='*', color=catches_color, label='Catches', markerfacecolor=catches_color, markersize=10)]
        ax.legend(handles=legend_elements, loc='upper right')
        # save and display fig
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def calc_prob_MOI_sequence(secsFromStart_dict, MOI, prevMOI):
    MOIProb_dict = {}
    for animal in secsFromStart_dict:
        MOIProb_dict[animal] = {}
        all_MOI = secsFromStart_dict[animal][MOI]
        all_prevMOI = secsFromStart_dict[animal][prevMOI]
        for day in range(len(all_MOI)):
            if len(all_MOI[day])>0:
                for moi in all_MOI[day]:
                    if moi in all_prevMOI[day]:
                        attempt_number = all_prevMOI[day].index(moi)
                        MOIProb_dict[animal][attempt_number] = MOIProb_dict[animal].setdefault(attempt_number,0) + 1
                    else:
                        try:
                            attempt_number = [x-moi>0 for x in all_prevMOI[day]].index(True)
                            MOIProb_dict[animal][attempt_number] = MOIProb_dict[animal].setdefault(attempt_number,0) + 1
                        except Exception:
                            print('Error in {a}, day {d}'.format(a=animal, d=day))
    return MOIProb_dict

def plot_probMOIseq(probMOIseq_dict, MOI_str, prevMOI_str, plots_dir, todays_dt):
    allA_probMOIseq = []
    for animal in probMOIseq_dict:
        all_numPrevMOI = sorted(probMOIseq_dict[animal].keys())
        if len(all_numPrevMOI)>0:
            max_numPrevMOI = max(all_numPrevMOI)
            frequencies_to_plot = [0*x for x in range(max_numPrevMOI+1)]
            for numPrevMOI in all_numPrevMOI:
                frequencies_to_plot[numPrevMOI] = probMOIseq_dict[animal][numPrevMOI]
        else:
            print('No previous MOIs for {a}'.format(a=animal))
        allA_probMOIseq.append(frequencies_to_plot)
    mostTS_beforeCatch = max([len(x) for x in allA_probMOIseq])
    for animal in range(len(allA_probMOIseq)):
        pad_N = mostTS_beforeCatch-len(allA_probMOIseq[animal])
        padded_animal = allA_probMOIseq[animal]+[0]*pad_N
        allA_probMOIseq[animal] = padded_animal
    # set figure save path and title
    figure_name = 'Prob_MomentsOfInterestSeq_' + prevMOI_str + 'Before' + MOI_str + '_allAnimals_' + todays_dt + '.png'
    figure_path = os.path.join(plots_dir, figure_name)
    figure_title = 'Number of '+ prevMOI_str + ' before a ' + MOI_str + ' for all animals'
    # set axes and other figure properties
    ax = plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    x = np.arange(mostTS_beforeCatch)  # the label locations
    plt.xticks(x)
    plt.xlabel("{pM} before {M}".format(pM=prevMOI_str, M=MOI_str))
    plt.ylabel("Frequency")
    plt.grid(b=True, which='major', linestyle='-')
    width = 0.1  # the width of the bars
    # draw bars for 5 animals at each x tick
    plt.bar(x - 2*width, allA_probMOIseq[0], width, label='L1-H2013-01')
    plt.bar(x - width, allA_probMOIseq[1], width, label='L1-H2013-02')
    plt.bar(x, allA_probMOIseq[2], width, label='L1-H2013-03')
    plt.bar(x + width, allA_probMOIseq[3], width, label='L7-H2013-01')
    plt.bar(x + 2*width, allA_probMOIseq[4], width, label='L7-H2013-02')
    # Add legend
    ax.legend()
    # create custom legend
    #legend_elements = [Line2D([0], [0], marker='|', color=food_offerings_color, label='Food offerings', markerfacecolor=food_offerings_color, markersize=10)]
    #ax.legend(handles=legend_elements, loc='upper right')
    # save and display fig
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
#raw_dataset_folder = r"C:\Users\Kampff_Lab\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw"
#plots_folder = r"C:\Users\Kampff_Lab\Documents\Github\CuttleShuttle-Analysis\Workflows\plots"

# List relevant data locations: these are for taunsquared
root_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows"
raw_dataset_folder = r"C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw"
plots_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows\plots"

# in raw dataset folder, list all csv files for moments of interest
animals = ['L1-H2013-01', 'L1-H2013-02', 'L1-H2013-03', 'L7-H2013-01', 'L7-H2013-02', 'L7-H2013-03']
allMOI_allA = {}
# extract data from csv and put into dictionary
print('Extracting raw data from csv...')
for animal in animals:
    print('Working on animal {a}'.format(a=animal))
    allMOI_allA[animal] = {}
    MOI_homebase = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "homebase*.csv")
    MOI_orients = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "orients*.csv")
    MOI_TS = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "tentacle_shots*.csv")
    MOI_catches = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "catches*.csv")
    food_offerings = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "food_available*.csv")
    session_vids = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "session_video*.csv")
    all_MOI = [MOI_homebase, MOI_orients, MOI_TS, MOI_catches, food_offerings, session_vids]
    for MOI_type in range(len(all_MOI)):
        for csv_file in all_MOI[MOI_type]:
            if os.path.getsize(csv_file)>0:
                csv_name = csv_file.split(os.sep)[-1]
                csv_date = csv_file.split(os.sep)[-2]
                csv_animal = csv_file.split(os.sep)[-3]
                current_dict_level = allMOI_allA[animal].setdefault(csv_date,{})
                # read csv file and convert timestamps into datetime objects
                str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8").split('+')[0][:-1], '%Y-%m-%dT%H:%M:%S.%f')
                csv_MOI = np.genfromtxt(csv_file, dtype=None, delimiter=",", converters={0:str2date})
                if MOI_type == 0:
                    allMOI_allA[animal][csv_date]['homebase'] = csv_MOI
                if MOI_type == 1:
                    allMOI_allA[animal][csv_date]['orients'] = csv_MOI
                if MOI_type == 2:
                    allMOI_allA[animal][csv_date]['tentacle shots'] = csv_MOI
                if MOI_type == 3:
                    allMOI_allA[animal][csv_date]['catches'] = csv_MOI
                if MOI_type == 4:
                    allMOI_allA[animal][csv_date]['food offerings'] = csv_MOI
                if MOI_type == 5:
                    allMOI_allA[animal][csv_date]['session vids'] = csv_MOI
print('Finished extracting csv data!')

# convert timestamp obj's into ints (microseconds from start)
MOIs = ['food offerings','homebase','orients','tentacle shots','catches']
print('Converting timestamps to microseconds from start...')
allMOI_allA_converted = convert_timestamps_to_secs_from_start(allMOI_allA, MOIs)
# convert animal numbers into names
animal_names = {'L1-H2013-01':'Dora','L1-H2013-02':'Scar','L1-H2013-03':'Ender','L7-H2013-01':'Old Tom','L7-H2013-02':'Plato','L7-H2013-03':'Blaise'}

########################################################
### ---- CREATE TIMELINE OF MOI FOR EACH ANIMAL ---- ###
########################################################

print('Drawing timeline of MOIs...')
plot_timeline_MOIs(allMOI_allA_converted, MOIs, animal_names, plots_folder, todays_datetime)

##################################################################################
### ---- PROBABILITY OF MOIS HAPPENING AFTER 1ST/2ND/3RD/4TH PREVIOUS MOI ---- ###
##################################################################################

print('Calculating probability of MOIs happening after previous MOI...')
# how many tentacle shots before a catch?
allA_shotsBeforeCatch = calc_prob_MOI_sequence(allMOI_allA_converted, 'catches', 'tentacle shots')
# accuracy: probability of catch after first shot?
first_shot_catch_prob = {}
second_shot_catch_prob = {}
third_shot_catch_prob = {}
for animal in allA_shotsBeforeCatch:
    if bool(allA_shotsBeforeCatch[animal]):
        total_shots = sum(allA_shotsBeforeCatch[animal].values())
        shots_without_error = allA_shotsBeforeCatch[animal].get(0, 0)
        first_shot_catch_prob[animal] = shots_without_error/total_shots
        two_attempts_or_less = allA_shotsBeforeCatch[animal].get(0, 0) + allA_shotsBeforeCatch[animal].get(1, 0)
        second_shot_catch_prob[animal] = two_attempts_or_less/total_shots
        three_attempts_or_less = allA_shotsBeforeCatch[animal].get(0, 0) + allA_shotsBeforeCatch[animal].get(1, 0) + allA_shotsBeforeCatch[animal].get(2, 0)
        third_shot_catch_prob[animal] = three_attempts_or_less/total_shots
mean_first_shot_catch = np.mean([x for x in first_shot_catch_prob.values()])
var_first_shot_catch = np.var([x for x in first_shot_catch_prob.values()])
mean_second_shot_catch = np.mean([x for x in second_shot_catch_prob.values()])
var_second_shot_catch = np.var([x for x in second_shot_catch_prob.values()])
mean_third_shot_catch = np.mean([x for x in third_shot_catch_prob.values()])
var_third_shot_catch = np.var([x for x in third_shot_catch_prob.values()])
# plot summary of catch accuracy
plot_probMOIseq(allA_shotsBeforeCatch, 'catch', 'tentacle shots', plots_folder, todays_datetime)
# how many orientations before a tentacle shot?
allA_orientsBeforeTS = calc_prob_MOI_sequence(allMOI_allA_converted, 'tentacle shots', 'orients')
plot_probMOIseq(allA_orientsBeforeTS, 'tentacle shot', 'orientations', plots_folder, todays_datetime)

###########################################################
### ---- FRAMES FROM TGB TO TENTACLES CONTACT PREY ---- ###
###########################################################
