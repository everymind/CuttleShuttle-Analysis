import os
import glob
import cv2
import datetime
import numpy as np
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
        all_mois = [all_food_offerings, all_homebases, all_orientations, all_tentacle_shots]
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

def plot_timeline_MOIs(dict_allAnimals_allMOIs, animal_names_dict, plots_dir, todays_dt):
    for animal in dict_allAnimals_allMOIs:
        all_session_lens = dict_allAnimals_allMOIs[animal]['session durations']
        all_session_dates = dict_allAnimals_allMOIs[animal]['session dates']
        all_homebases = dict_allAnimals_allMOIs[animal]['homebase']
        all_orientations = dict_allAnimals_allMOIs[animal]['orients']
        all_tentacle_shots = dict_allAnimals_allMOIs[animal]['tentacle shots']
        all_food_offerings = dict_allAnimals_allMOIs[animal]['food offerings']
        # remove habituation session
        #    all_session_dates = all_session_dates[1:]
        #    all_session_lens = all_session_lens[1:]
        #    all_homebases = all_homebases[1:]
        #    all_orientations = all_orientations[1:]
        #    all_tentacle_shots = all_tentacle_shots[1:]
        #    all_food_offerings = all_food_offerings[1:]
        # reverse lists of MOI times so that first session appears at the top of the plot
        all_session_lens_reversed = all_session_lens[::-1]
        all_session_dates_reversed = all_session_dates[::-1]
        all_homebases_reversed = all_homebases[::-1]
        all_orientations_reversed = all_orientations[::-1]
        all_tentacle_shots_reversed = all_tentacle_shots[::-1]
        all_food_offerings_reversed = all_food_offerings[::-1]
        session_len_mins = 37
        plotting_session_len = session_len_mins*60
        # find number of each MOI
        hb_count = sum([len(session) for session in all_homebases_reversed])
        orients_count = sum([len(session) for session in all_orientations_reversed])
        ts_count = sum([len(session) for session in all_tentacle_shots_reversed])
        fo_count = sum([len(session) for session in all_food_offerings_reversed])

        homebase_color = [1.0, 0.0, 0.0, 0.5]
        orientations_color = [0.0, 1.0, 0.0, 0.5]
        tentacle_shots_color = [0.0, 0.0, 1.0, 0.5]
        food_offerings_color = [1.0, 0.0, 1.0, 0.5]

        figure_name = 'MomentsOfInterest_'+ animal + '_' + todays_dt + '.png'
        figure_path = os.path.join(plots_dir, figure_name)
        figure_title = "Moments of interest during hunting session of " + animal + ", aka " + animal_names_dict[animal] + "\nReturns to home base: " + str(hb_count) + "\n Number of Orientations: " + str(orients_count) + "\n Number of Tentacle Shots: " + str(ts_count)

        ax = plt.figure(figsize=(16,9), dpi=200)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        plt.xlim(-60,plotting_session_len+60)
        plt.ylim(-1, len(all_session_lens)+0.5)
        plot_yticks = np.arange(0, len(all_session_lens), 1)
        ytick_labels = ['Day '+str(x) for x in range(len(all_session_dates_reversed))]
        plt.yticks(plot_yticks, ytick_labels[::-1])
        plt.ylabel("Session Number")
        plot_xticks = np.arange(0, plotting_session_len, step=60)
        plt.xticks(plot_xticks, ['%d'%(x/60) for x in plot_xticks])
        plt.xlabel("Minute from start of hunting session")
        #plt.xlabel("Frame number, original framerate = 60fps")
        plt.grid(b=True, which='major', linestyle='-')

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

        legend_elements = [Line2D([0], [0], marker='|', color=food_offerings_color, label='Food offerings', markerfacecolor=food_offerings_color, markersize=10),
                        Line2D([0], [0], marker='d', color=homebase_color, label='Returns to homebase', markerfacecolor=homebase_color, markersize=10),
                        Line2D([0], [0], marker='^', color=orientations_color, label='Orientations', markerfacecolor=orientations_color, markersize=10),
                        Line2D([0], [0], marker='*', color=tentacle_shots_color, label='Tentacle Shots', markerfacecolor=tentacle_shots_color, markersize=10)]
        ax.legend(handles=legend_elements, loc='upper right')

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
    allMOI_allA[animal] = {}
    MOI_homebase = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "homebase*.csv")
    MOI_orients = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "orients*.csv")
    MOI_TS = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "tentacle_shots*.csv")
    food_offerings = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "food_available*.csv")
    session_vids = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "session_video*.csv")
    all_MOI = [MOI_homebase, MOI_orients, MOI_TS, food_offerings, session_vids]
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
                    allMOI_allA[animal][csv_date]['food offerings'] = csv_MOI
                if MOI_type == 4:
                    allMOI_allA[animal][csv_date]['session vids'] = csv_MOI

########################################################
### ---- CREATE TIMELINE OF MOI FOR EACH ANIMAL ---- ###
########################################################

MOIs = ['food offerings', 'homebase','orients','tentacle shots']
print('Converting timestamps to seconds from start...')
allMOI_allA_converted = convert_timestamps_to_secs_from_start(allMOI_allA, MOIs)
animal_names = {'L1-H2013-01':'Dora','L1-H2013-02':'Scar','L1-H2013-03':'Ender','L7-H2013-01':'Old Tom','L7-H2013-02':'Plato','L7-H2013-03':'Blaise'}
print('Plotting...')
plot_timeline_MOIs(allMOI_allA_converted, animal_names, plots_folder, todays_datetime)

