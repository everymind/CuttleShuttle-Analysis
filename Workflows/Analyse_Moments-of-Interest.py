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
from scipy import stats

### FUNCTIONS ###

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
animals = ['L1-H2013-01', 'L1-H2013-02', 'L1-H2013-03', 'L7-H2013-01', 'L7-H2013-02']
allMOI_allA = {}
for animal in animals:
    allMOI_allA[animal] = {}
    MOI_homebase = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "homebase*.csv")
    MOI_orients = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "orients*.csv")
    MOI_TS = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "tentacle_shots*.csv")
    session_vids = glob.glob(raw_dataset_folder + os.sep + animal + os.sep + "*" + os.sep + "session_video*.csv")
    all_MOI = [MOI_homebase, MOI_orients, MOI_TS, session_vids]
    for MOI_type in range(len(all_MOI)):
        for csv_file in all_MOI[MOI_type]:
            if os.path.getsize(csv_file)>0:
                csv_name = csv_file.split(os.sep)[-1]
                csv_date = csv_file.split(os.sep)[-2]
                csv_animal = csv_file.split(os.sep)[-3]
                # read csv file and convert timestamps into datetime objects
                str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8").split('+')[0][:-1], '%Y-%m-%dT%H:%M:%S.%f')
                csv_MOI = np.genfromtxt(csv_file, dtype=None, delimiter=",", converters={0:str2date})
                if MOI_type == 0:
                    allMOI_allA[animal].setdefault(csv_date,{}).setdefault('homebase', []).append(csv_MOI)
                if MOI_type == 1:
                    allMOI_allA[animal].setdefault(csv_date,{}).setdefault('orients', []).append(csv_MOI)
                if MOI_type == 2:
                    allMOI_allA[animal].setdefault(csv_date,{}).setdefault('tentacle shots', []).append(csv_MOI)
                if MOI_type == 3:
                    allMOI_allA[animal].setdefault(csv_date,{}).setdefault('session vids', []).append(csv_MOI)

########################################################
### ---- CREATE TIMELINE OF MOI FOR EACH ANIMAL ---- ###
########################################################

MOIs = ['homebase','orients','tentacle shots']

for animal in allMOI_allA:
    all_session_lens = []
    all_homebases = []
    all_orientations = []
    all_tentacle_shots = []
    all_mois = [all_homebases, all_orientations, all_tentacle_shots]
    for session_date in sorted(allMOI_allA[animal].keys()):
        if len(allMOI_allA[animal][session_date].keys())==1 and 'session vids' in allMOI_allA[animal][session_date]:
            print("No moments of interest for animal {a} on {s}".format(a=animal,s=session_date))
        else:
            start_ts = allMOI_allA[animal][session_date]['session vids'][0][0]
            end_ts = allMOI_allA[animal][session_date]['session vids'][0][-1]
            session_len_td = end_ts - start_ts
            session_len = session_len_td.seconds + session_len_td.microseconds/1000
            all_session_lens.append(session_len)
            for moi in range(len(MOIs)):
                if MOIs[moi] in allMOI_allA[animal][session_date]:
                    this_session_mois = []
                    if len(allMOI_allA[animal][session_date][MOIs[moi]][0]) <= 1:
                        timestamp = allMOI_allA[animal][session_date][MOIs[moi]][0]
                        time_diff_dt = timestamp - start_ts
                        time_diff = time_diff_dt.seconds + time_diff_dt.microseconds/1000
                        this_session_mois.append(time_diff)
                    else:
                        for timestamp in allMOI_allA[animal][session_date][MOIs[moi]][0]:
                            time_diff_dt = timestamp - start_ts
                            time_diff = time_diff_dt.seconds + time_diff_dt.microseconds/1000
                            this_session_mois.append(time_diff)
                    all_mois[moi].append(this_session_mois)

            
            
    figure_name = 'MomentsOfInterest_'+ animal + '_' + todays_dt + '.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = "Moments of interest during hunting session of " + animal + "\nReturns to home base: " + str(len(allMOI_allA[animal][session_date]['homebase'][0])) + "\n Number of Orientations: " + str(len(allMOI_allA[animal][session_date]['orients'][0])) + "\n Number of Tentacle Shots: " + str(len(allMOI_allA[animal][session_date]['tentacle shots'][0]))
    
    plt.figure(figsize=(16,9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    plt.xlim(0,session_len)
    plt.ylim(-0.5, 3.5)
    plt.ylabel("Moment of Interest")
    plot_xticks = np.arange(0, session_len, step=60)
    plt.xticks(plot_xticks, ['%.1f'%(x/60) for x in plot_xticks])
    plt.xlabel("Minutes")
    #plt.xlabel("Frame number, original framerate = 60fps")
    plt.grid(b=True, which='major', linestyle='-')
    
    plt.plot(time_diff, 1, Marker='o', color=[1.0, 0.0, 0.0, 0.8])
    plt.plot(time_diff, 2, Marker='^', color=[0.0, 1.0, 0.0, 0.8])
    plt.plot(time_diff, 3, Marker='*', color=[0.0, 0.0, 1.0, 0.8])

        

