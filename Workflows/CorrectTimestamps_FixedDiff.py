import os
import glob
import cv2
import datetime
import numpy as np
import sys
import csv

# WHEN RUNNING AS SCRIPT IN TERMINAL #
inMOIFile = sys.argv[1]
inSessionVidFile = sys.argv[2]
targetFrame = int(sys.argv[3])
outMOIFilePath = sys.argv[4]
# WHEN DEBUGGING #
#inMOIFile = r"C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw\L1-H2013-01\2014-09-09\BuggyOriginal_food_available2014-09-09.csv"
#inSessionVidFile = r"C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw\L1-H2013-01\2014-09-09\session_video2014-09-09T15_14_08.csv"
#targetFrame = 4022
#outMOIFilePath = r"C:\Users\taunsquared\Dropbox\CuttleShuttle\CuttleShuttle-VideoDataset-Raw\L1-H2013-01\2014-09-09\food_available_updated_2014-09-09.csv"

# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()

# List relevant data locations: these are for KAMPFF-LAB
#root_folder = r"C:\Users\Kampff_Lab\Documents\Github\CuttleShuttle-Analysis\Workflows"

# List relevant data locations: these are for taunsquared
root_folder = r"C:\Users\taunsquared\Documents\GitHub\CuttleShuttle-Analysis\Workflows"

# extract data from input csv files
print('Extracting data from input CSV files...')
str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8").split('+')[0][:-1], '%Y-%m-%dT%H:%M:%S.%f')
timestamps_MOI = np.genfromtxt(inMOIFile, dtype=None, delimiter=",", converters={0:str2date})
timestamps_SessionVid = np.genfromtxt(inSessionVidFile, dtype=None, delimiter=",", converters={0:str2date})
timestamps = [timestamps_MOI, timestamps_SessionVid]

### BEGIN CONVERSION ###
# grab first MOI timestamp from inMOIFile
firstTimestamp_MOIstart = timestamps_MOI[0]
# grab first MOI timestamp from input targetFrame
targetTimestamp_MOIstart = timestamps_SessionVid[targetFrame-1]
# calculate diff
time_diff = firstTimestamp_MOIstart - targetTimestamp_MOIstart
# apply diff to all timestamps in input MOI csv and save to output MOI csv
corrected_MOI_timestamps  = open(outMOIFilePath, 'w', newline='')
with corrected_MOI_timestamps:
    timestamp_writer = csv.writer(corrected_MOI_timestamps, dialect='excel', delimiter=',')
    corrected_timestamps = []
    for timestamp in timestamps_MOI:
        corrected_TS = timestamp - time_diff
        corrected_TS_utf8 = corrected_TS.strftime('%Y-%m-%dT%H:%M:%S.%f  ')
        corrected_timestamps.append([corrected_TS_utf8])
    corrected_timestamps = np.array(corrected_timestamps)
    timestamp_writer.writerows(corrected_timestamps)

print('Finished writing corrected timestamps to file!')