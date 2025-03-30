import time
import cv2
import sys
import numpy
import re
import os
import math
import trackpy
import trackpy.predict

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from scipy.optimize import curve_fit
import cupy
import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.exporters
from PyQt5 import  QtWidgets 
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QDialog,
                             QPushButton,  QFileDialog, QPlainTextEdit, QCheckBox, 
                             QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, 
                             QMenuBar, QMenu, QAction, QMessageBox, QPlainTextEdit)
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QFont
global maxLength, reconnectStops, displayAllStops # maxLength = length of the longest route #reconnectStops = boolean for the function that reconnects interrupted stops
global DURATION_MEASURE_SPEED, MIN_SPEED_FIT, STOP_THRESHOLD_X, STOP_THRESHOLD_Y, STOP_THRESHOLD_TEMP, DURATION_CHECK_STOP, DELAY_CHECK_SPEED, BACKWARD_RETURN_THRESHOLD, MAX_MOVEMENT_THRESHOLD_STOP_X, MAX_MOVEMENT_THRESHOLD_STOP_Y, MAX_TEMP_SEP_RECONNECT_STOP, COEFF_AREA_PLUS, COEFF_AREA_MINUS, COEFF_MaxVIncrease, COEFF_MinVDecrease, MAX_AREA_HISTO, STEP_HISTO_AREAS, STEP_HISTO_SPEEDS, MIN_DISTANCE_TWO_BALLS_X, MIN_DISTANCE_TWO_BALLS_Y, TEMP_CONVERSION_FACTOR, PIXEL_MICRON_CONVERSION_FACTOR, CLICK_PROXIMITY_THRESHOLD_X, CLICK_PROXIMITY_THRESHOLD_T
#####constants #####
#####Trajectories constants #####
THRESHOLD_MEAN = 60 # luminance threshold for video segmentation

DEFAULT_PERCENTILE = 90 # default luminance percentile # NOTE: high value necessary (percentile, not luminance percentage)
DEFAULT_MASS_MIN = 1 # minimum luminance mass # NOTE: value 1 seems necessary
DEFAULT_DIMENSION_X = 20 # default value for the X dimension of the beads, rounded and always odd # must be slightly greater than the diameter of the beads
DEFAULT_DIMENSION_Y = 10 # default value for the Y dimension of the beads, rounded and always odd # must be slightly greater than the diameter of the beads
DEFAULT_SEARCH_RANGE_X = 16 # maximum default value for movement in X, must be slightly greater than the maximum speed of the beads
DEFAULT_SEARCH_RANGE_Y = 2 # maximum default value for movement in Y
DEFAULT_SEPARATION_X = 10
DEFAULT_SEPARATION_Y = 5
DEFAULT_MEMORY = 3 # maximum number of images for which a bead can disappear and be linked to the same trajectory
DEFAULT_THRESHOLD = 25 # default threshold number of images for which a trajectory is recorded  

PERCENTILE = DEFAULT_PERCENTILE
MIN_MASS = DEFAULT_MASS_MIN
DIM_X = DEFAULT_DIMENSION_X
DIM_Y = DEFAULT_DIMENSION_Y
SEARCH_INTERVAL_X = DEFAULT_SEARCH_RANGE_X
SEARCH_INTERVAL_Y = DEFAULT_SEARCH_RANGE_Y
SEPARATION_X = DEFAULT_SEPARATION_X
SEPARATION_Y = DEFAULT_SEPARATION_Y
MEMORY = DEFAULT_MEMORY
THRESHOLD = DEFAULT_THRESHOLD

##### Stop Constants #####
DEFAULT_DURATION_MEASURE_SPEED = 4 # in camera frame duration
DEFAULT_MIN_SPEED_FIT = 0.3 # minimum speed threshold of the histogram during Gaussian fit creation # in pixels/frame
# stationary conditions
DEFAULT_STOP_THRESHOLD_X = 0.5 # spatial threshold in X for defining stops # in pixels
DEFAULT_STOP_THRESHOLD_Y = 10 # spatial threshold in Y for defining stops # in pixels, high value because not used
DEFAULT_STOP_THRESHOLD_TEMP = 4   # temporal threshold for defining stops # in camera frame duration
DEFAULT_STOP_THRESHOLD_TEMP_OUT_TRI = 1
DEFAULT_DURATION_CHECK_STOP = 3 # points to check stop conditions during start of stop (= during DEFAULT_STOP_THRESHOLD_TEMP)
# conditions around stop speed, area, ...
DEFAULT_DELAY_CHECK_AROUND = 5  # reserved for post-stop tests
DEFAULT_DELAY_CHECK_SPEED = 5 # delay between speed verification points and start of stop
DEFAULT_DURATION_CHECK_AROUND_SPEED = 4 # duration of checking stops for speed before stop (= first points of DEFAULT_DELAY_CHECK_AROUND)
DEFAULT_DURATION_CHECK_AROUND_AREA = 3 # duration of checking stops for area before stop (= last points of DEFAULT_DELAY_CHECK_AROUND)
DEFAULT_DURATION_MEASURE_RETURN = 6
DEFAULT_BACKWARD_RETURN_THRESHOLD = -0.8 # max distance in pixels of an artifact slowing point from the stop position in X
DEFAULT_MAX_MOVEMENT_THRESHOLD_STOP_X = 2 # max distance in X traveled during a stop // 2 necessary to compensate global drift
DEFAULT_MAX_MOVEMENT_THRESHOLD_STOP_Y = 1 # max movement of a stop in Y (used to reconnect interrupted stops)
DEFAULT_DELAY_SHIFT_START_STOP_OUT_TRI = 5 # shift towards stabilized part of a start of stop outside of sorting
DEFAULT_MAX_TEMP_SEP_RECONNECT_STOP = 10000 # max duration between two fragments of stops to reconnect, in frames
DEFAULT_COEFF_AREA_PLUS = 1.6 # coefficient for increasing/decreasing limit areas
DEFAULT_COEFF_AREA_MINUS = 0.8
DEFAULT_COEFF_MaxVIncrease = 1.5 # coefficient for increasing/decreasing limit speeds
DEFAULT_COEFF_MinVDecrease = 1
DEFAULT_MAX_AREA_HISTO = 1000 # in pixels²
DEFAULT_STEP_HISTO_AREAS = 2 # in pixels²
DEFAULT_STEP_HISTO_SPEEDS = 0.05 # in pixels/frame
DEFAULT_MIN_DISTANCE_TWO_BEADS_X = 18  # in pixels, minimum distance between the centroids of two balls to ensure no contact during stop
DEFAULT_MIN_DISTANCE_TWO_BEADS_Y = 9  # in pixels, minimum distance between the centroids of two balls to ensure no contact during stop
DEFAULT_TEMP_CONVERSION_FACTOR = 0.02 # in seconds/frame
DEFAULT_PIX_MICRON_CONVERSION_FACTOR = 0.331 # in µm/pixel
DEFAULT_CLICK_PROXIMITY_THRESHOLD_X = 10 # for manual sorting of stops
DEFAULT_CLICK_PROXIMITY_THRESHOLD_T = 10 # for manual sorting of stops

DURATION_MEASURE_SPEED = DEFAULT_DURATION_MEASURE_SPEED
MIN_SPEED_FIT = DEFAULT_MIN_SPEED_FIT
STOP_THRESHOLD_X = DEFAULT_STOP_THRESHOLD_X
STOP_THRESHOLD_Y = DEFAULT_STOP_THRESHOLD_Y
STOP_THRESHOLD_TEMP = DEFAULT_STOP_THRESHOLD_TEMP
STOP_THRESHOLD_TEMP_OUT_TRI = DEFAULT_STOP_THRESHOLD_TEMP_OUT_TRI
DURATION_CHECK_STOP = DEFAULT_DURATION_CHECK_STOP
DELAY_CHECK_AROUND = DEFAULT_DELAY_CHECK_AROUND
DELAY_CHECK_SPEED = DEFAULT_DELAY_CHECK_SPEED
DURATION_MEASURE_RETURN = DEFAULT_DURATION_MEASURE_RETURN
DURATION_CHECK_AROUND_SPEED = DEFAULT_DURATION_CHECK_AROUND_SPEED 
DURATION_CHECK_AROUND_AREA = DEFAULT_DURATION_CHECK_AROUND_AREA 
BACKWARD_RETURN_THRESHOLD = DEFAULT_BACKWARD_RETURN_THRESHOLD
MAX_MOVEMENT_THRESHOLD_STOP_X = DEFAULT_MAX_MOVEMENT_THRESHOLD_STOP_X
MAX_MOVEMENT_THRESHOLD_STOP_Y = DEFAULT_MAX_MOVEMENT_THRESHOLD_STOP_Y
DELAY_SHIFT_START_STOP_OUT_TRI = DEFAULT_DELAY_SHIFT_START_STOP_OUT_TRI
MAX_TEMP_SEP_RECONNECT_STOP = DEFAULT_MAX_TEMP_SEP_RECONNECT_STOP
COEFF_AREA_PLUS = DEFAULT_COEFF_AREA_PLUS
COEFF_AREA_MINUS = DEFAULT_COEFF_AREA_MINUS
COEFF_MaxVIncrease = DEFAULT_COEFF_MaxVIncrease
COEFF_MinVDecrease = DEFAULT_COEFF_MinVDecrease
MAX_AREA_HISTO = DEFAULT_MAX_AREA_HISTO
STEP_HISTO_AREAS = DEFAULT_STEP_HISTO_AREAS
STEP_HISTO_SPEEDS = DEFAULT_STEP_HISTO_SPEEDS
MIN_DISTANCE_TWO_BEADS_X = DEFAULT_MIN_DISTANCE_TWO_BALLS_X
MIN_DISTANCE_TWO_BEADS_Y = DEFAULT_MIN_DISTANCE_TWO_BALLS_Y
TEMP_CONVERSION_FACTOR = DEFAULT_TEMP_CONVERSION_FACTOR
PIX_MICRON_CONVERSION_FACTOR = DEFAULT_PIX_MICRON_CONVERSION_FACTOR
CLICK_PROXIMITY_THRESHOLD_X = DEFAULT_CLICK_PROXIMITY_THRESHOLD_X
CLICK_PROXIMITY_THRESHOLD_T = DEFAULT_CLICK_PROXIMITY_THRESHOLD_T

SINGLE_GAUSSIAN_AREA = True # for areas
TWO_GAUSSIANS_AREA = False # for areas

reconnectStops = True # boolean for reconnecting interrupted stops

displayAllStops = False



##### Trajectories functions  #####

def segmentVideo(address, threshold): # Determine intervals of real acquisition in the complete film
    # Read the images without saving the data of each (without arrays)
    Start = time.time()
    startAcquisitions = []
    endAcquisitions = []     
    # addressSplit = os.path.split(address)    # For saving address composition # Separate folder path from file name      
    capture = cv2.VideoCapture(address)  
    wasPreviousFrameDark = False
    frameNumber = 0
    try: 
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            # frameNetB = frame[:,:,0] # Keep luminance on R of RGB
            partialFrameMean = cupy.mean(frame[:100, :100, 0]) # Mean over a sub-part of 100 * 100 pixels² of the image, using only one color
            if partialFrameMean < threshold:
                if not wasPreviousFrameDark and frameNumber > 0: # Case of darkening after a bright sequence
                    endAcquisitions.append(frameNumber)
                    print("end")
                    print(frameNumber)
                wasPreviousFrameDark = True
            else: # Case of brightness
                if wasPreviousFrameDark or frameNumber == 0: # Case of start of a bright sequence or start of a bright film
                    startAcquisitions.append(frameNumber)
                    print("start")
                    print(frameNumber)
                wasPreviousFrameDark = False
            frameNumber += 1                  
    except:
        print("Error opening film")
    finally:
        capture.release()
    if wasPreviousFrameDark == False:  # Case of end of film during a bright sequence
        endAcquisitions.append(frameNumber - 1)  # -1 because the frame number increment follows the reading of the frame in previous cases  
        print("end")
        print(frameNumber)
    End = time.time()
    print(End - Start)
    return (startAcquisitions, endAcquisitions)

def createBoundedPairs(acquisitions): # creation of start-end pairs with boundaries obtained from the fractionneVideo function
    acquisitionList = []
    startAcquisitions = acquisitions[0]
    endAcquisitions = acquisitions[1]
    if len(startAcquisitions) == len(endAcquisitions):
        for n in range(len(startAcquisitions)):        
            acqu = (startAcquisitions[n], endAcquisitions[n])
            acquisitionList.append(acqu)
    else:
        print("Problem determining acquisition boundaries")
    print('Division into sequences completed')    
    # print(acquisitionList)
    return acquisitionList


def processAcquisition(address, acquisition):   # Detects particles and creates trajectories
    capture = cv2.VideoCapture(address)    
    # addressSplit = os.path.split(address)    # For saving address composition and separating folder path from file name      
    frames = []
    for numFrame in range(acquisition[0], acquisition[1]):
        capture.set(cv2.CAP_PROP_POS_FRAMES, numFrame - 1)
        ret, frame = capture.read()        
        if not ret:
            break
        frameNetB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frameNetBInverse = 255 - frameNetB  # Invert the image        
        frames.append(frameNetBInverse)
    capture.release()    
    frames = numpy.array(frames)
    
    # Particle detection
    start = time.time()
    particles = []    
    dim_X = DIM_X if DIM_X % 2 == 1 else DIM_X + 1  # For format: tp.locate requires odd values
    dim_Y = DIM_Y if DIM_Y % 2 == 1 else DIM_Y + 1  # For format: tp.locate requires odd values
    dimension = [dim_Y, dim_X]  # Dimensions for bead prediction: major axis (DIM_X) and minor axis (DIM_Y) of the ellipse
    
    particles = trackpy.batch(frames, dimension, percentile=PERC, minmass=MINMASSE, separation=(SEPARATION_X, SEPARATION_Y), processes=11)  # DataFrame([x, y, mass, size, ecc, signal, raw_mass])
    
    # Creating trajectories
    particles.reset_index(drop=True, inplace=True)  # Drop the index since one of the columns describes the frame numbers
    searchIntervalY = float(INTERVALLE_RECH_Y)
    searchIntervalX = float(INTERVALLE_RECH_X)
    searchInterval2D = [searchIntervalY, searchIntervalX]
    
    @trackpy.predict.predictor  # WARNING: Not used with simple velocity vector, as it requires favoring detection of beads that have been stationary for a long time 
    def predis(t1, particle):
        velocity = numpy.array((0, -8))  # (v_y, v_x)
        return particle.pos + velocity * (t1 - particle.t)
    
    t = trackpy.link_df(particles, searchInterval2D, memory=MEMOIRE)  # Use trackpy.link to link the detected particles in the images
    
    # Preparing data for text output
    t1 = trackpy.filter_stubs(t, SEUIL)  # Function of trackpy that eliminates trajectories shorter than a certain "THRESHOLD"
    t1 = t1.reset_index(drop=True)
    # tSorted = t1.copy()  # To keep t1 (if testing) 
    t1['x'] = t1['x'].apply(lambda x: round(x, 2))  # Reduce number of decimals 
    t1['y'] = t1['y'].apply(lambda x: round(x, 2))  # Reduce number of decimals
    tSorted = t1.sort_values(['particle', 'frame'])  # Sort by particle index in ascending order first and then if there are rows with the same 'particle' value, they will be sorted by 'frame' in ascending order
    # textCheck = tSorted.to_string()
    # outputFile = open("C:\\Users\\Philippe\\Desktop\\test.txt","w")
    # outputFile.write(textCheck)     
    # #### resultTable = tSorted.to_numpy(copy=True)  # shape = nimages * 12 columns # complete resultTable is used for saving trajectories in .txt 
    resultTable = cupy.asarray(tSorted.values)  # shape = nimages * 12 columns 
    changeTraj = resultTable[:-1, 11] - resultTable[1:, 11] != 0  # The 12th column is the trajectory number, the difference between 2 rows indicates a change of trajectory
    posChangeTraj = changeTraj.nonzero()[0]  # Retrieve positions
    numberTrajs = posChangeTraj.shape[0] + 1
    # ##### resultTable = numpy.delete(resultTable, (2, 3, 5, 6, 7, 8, 9, 11), axis=1)  # Keep columns y, x, diameter in x, frame number
    resultTable = resultTable[:, [10, 1, 0, 4]]  # cupy.delete doesn't work # Remove columns 2, 3, 5, 6, 7, 8, 9, 11; keep columns y, x, diameter

    # ##### resultTable = numpy.delete(resultTable, (2, 3, 5, 6, 7, 8, 9, 11), axis=1)  # Keep columns y, x, diameter in x, frame number
    resultTable = resultTable[:, [10, 1, 0, 4]]  # cupy.delete doesn't work # Remove columns 2, 3, 5, 6, 7, 8, 9, 11; keep columns y, x, diameter in x, frame number and rearrange columns for    numFrame, x, y, diameter
    resultTable[:, 0] = resultTable[:, 0] + acquisition[0]  # Convert to absolute time in the film 
    # ##### resultTable[:, [0, 1, 2, 3]] = resultTable[:, [3, 1, 0, 2]]  # Rearrange columns for numFrame, x, y, diameter
    resultTable[:, 3] = resultTable[:, 3] / 2 * resultTable[:, 3] / 2 * cupy.pi  # Convert to area  
    # !!! Difference with pyTrajs
    listOfSubTables = []  # For direct transformation into a table read by the stop detection functions via creeTableTrajsInterne
    # End of difference with pyTrajs
    for n, ch in enumerate(posChangeTraj):        
         # !!! Difference with pyTrajs
         if n == 0:
             currentSubTable = resultTable[: ch + 1, :]
        else:
             currentSubTable = resultTable[posChangeTraj[n - 1] + 1: ch + 1, :]
             listOfSubTables.append(currentSubTable)   
     if len(posChangeTraj) > 0:
         listOfSubTables.append(resultTable[ch + 1:, :])  # End of iteration after last trajectory change
     # End of difference with pyTrajs      
     End = time.time()
     print(End - Start)
     return listOfSubTables, numberTrajs  # Difference with pyTrajs: all other tracking results (eccentricity, ...) are available in t1 (= summary table)


def saveTextResults(subTablesList, filePath): # Saves trajectories as .txt files for potential reanalysis
    outputFolderPath = os.path.dirname(filePath)
    if not os.path.exists(outputFolderPath):
        os.makedirs(outputFolderPath)
    outputFile = open(filePath, "w")
    outputFile.write("")    
    separator = "-1.0\t0.0\t0.0\t0.0\n"
    for table in subTablesList:
        tempFilePath = filePath[len(filePath) - 4:] + "temp.txt"
        cupy.savetxt(tempFilePath, table, fmt='%.2f', delimiter='\t', newline='\n') 
        tempFile = open(tempFilePath, "r") 
        for line in tempFile: 
            outputFile.write(line)         
        outputFile.write(separator) 
        tempFile.close()
        os.remove(tempFilePath)

############### Arrests functions ################

############ Mathematical functions for fits ##########

def gaussian(x, a, x0, sigma):  # definition of fit function for speed and area histograms # x, amplitude, mean, standard deviation
    return a * numpy.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def mix2Gaussians(x, a, x0, sigma, b, xb0, sigmab):  # definition of fit function for speed and area histograms # x, amplitude, mean, standard deviation
    return gaussian(x, a, x0, sigma) + gaussian(x, b, xb0, sigmab)

############ Functions for creating numpy table of trajectories in the form: nTrajs * nTypesData * nPointsofTrajs ##########

def createInternalTrajectoryTable(subTablesList):  # Direct transformation of detected trajectories by film processing functions and formation of numpy table of trajectories in the form: nTrajs * nTypesData * nPointsofTrajs
    subTables = subTablesList
    # print("long list of subTables")
    # print(len(subTables))
    # for n in range(len(subTables)):
    #     print(subTables[n].shape)
    subTablesForStartingFrame = subTables[0:-1] 
    subTablesForStartingFrame.sort(key=lambda subTable: subTable[0][0])  # Sort by smallest time # difference sort and sorted
    startingFrame = subTablesForStartingFrame[0][0][1]  # OK        
    subTables.sort(key=lambda subTable: subTable.shape[0], reverse=True)  # Sort by decreasing length of trajectories
    subTables[0] = numpy.insert(subTables[0].get(), subTables[0].get().shape[0] - 1, 0, axis=0)  # Add a row of zeros for NaN and non-connection in PyQTgraph  # Convert from numpy to cupy
    subTables[0] = cupy.array(subTables[0]) 
    maxLength = subTables[0].shape[0]
    constantLengthTrajectoriesList = []  # List to construct trajectories of constant length
    subTables[0] = cupy.transpose(subTables[0], (1, 0))  # Form first element of new table
    # subTables[0][0, :] = subTables[0][0, :] + subTables[0][1, -1]  # Convert to absolute time # unnecessary with pyTrajs
    subTables[0] = subTables[0][:, :-1]  # Remove row separating between trajectories from the first element
    constantLengthTrajectoriesList.append(subTables[0])  
    # @njit
    def shapeTrajectoriesNP(subTables, constantLengthTrajectoriesList):  # Ensure all sub-tables are of constant length 
        for subTable in subTables[1:]:          
            if subTable.shape[0] > (3 * DURATION_MEASURE_SPEED + 2 * DELAY_CHECK_AROUND):  # Minimum duration threshold for stop checks    
                subTable = cupy.transpose(subTable, (1, 0))  # Shape: nTypesData * nPointsofTrajs         
                # subTable[0, :] = subTable[0, :] + subTable[1, -1]  # Convert to absolute time # unnecessary with pyTrajs
                subTable = subTable[:, :-1]  # Remove separating line
                subTableLongConst = cupy.concatenate((subTable, cupy.zeros((4, maxLength - subTable.shape[1] - 1))), axis=1)  # Set all trajectories to the same length by adding a table of NaN
                constantLengthTrajectoriesList.append(subTableLongConst)  
    shapeTrajectoriesNP(subTables, constantLengthTrajectoriesList)
    Trajs1Manip = cupy.array(constantLengthTrajectoriesList)  # Place arrays in columns in the form: nTrajs * nTypesData * nPointsofTrajs
    Trajs1Manip = cupy.where(Trajs1Manip[:, :, 1:] == 0, cupy.nan, Trajs1Manip[:, :, 1:])  # Use where to replace values # [,:,1:] avoids the first point of each traj which can be the first zero of the time later transformed into NaN
    return Trajs1Manip, startingFrame

def openTrajectoryFile(address):  # Open saved .txt trajectory files and create a numpy table of trajectories in the form: nTrajs * nTypesData * nPointsOfTrajs
    global maxLength
    trajs2D = cupy.loadtxt(address, dtype=cupy.float64, delimiter='\t') 
    # print(trajs2D)
    listMinusOne = (cupy.where(trajs2D == -1)[0] + 1).tolist()  # numpy.split requires a list, not np.array; offset by 1 because otherwise split before the line starting with -1
    subTables = cupy.split(trajs2D, listMinusOne, axis=0)  # numpy.split returns a list of np.arrays
    subTablesForStartingFrame = subTables[0:-1] 
    subTablesForStartingFrame.sort(key=lambda subTable: subTable[0][0])  # Sort by the smallest time; note the difference between sort and sorted
    startingFrame = subTablesForStartingFrame[0][0][1]  # OK        
    subTables.sort(key=lambda subTable: subTable.shape[0], reverse=True)  # Sort by decreasing length of trajectories
    subTables[0] = numpy.insert(subTables[0].get(), subTables[0].get().shape[0] - 1, 0, axis=0)  # Add a row of zeros for NaN and non-connection in PyQTgraph; convert from numpy to cupy
    subTables[0] = cupy.array(subTables[0]) 
    maxLength = subTables[0].shape[0]
    constantLengthTrajectoriesList = []  # List to build constant length trajectories
    subTables[0] = cupy.transpose(subTables[0], (1, 0))  # Form the first element of the new table
    # subTables[0][0, :] = subTables[0][0, :] + subTables[0][1, -1]  # Convert to absolute time; unnecessary with pyTrajs
    subTables[0] = subTables[0][:, :-1]  # Remove the separating line between trajectories from the first element
    constantLengthTrajectoriesList.append(subTables[0])

    def shapeTrajectoriesNP(subTables, constantLengthTrajectoriesList):
        for subTable in subTables[1:]:          
            if subTable.shape[0] > (3 * DURATION_MEASURE_SPEED + 2 * DELAY_CHECK_AROUND):  # Minimum duration threshold for stop checks    
                subTable = cupy.transpose(subTable, (1, 0))  # Shape: nTypesData * nPointsOfTrajs         
                # subTable[0, :] = subTable[0, :] + subTable[1, -1]  # Convert to absolute time
                subTable = subTable[:, :-1]  # Remove separating line -1.0, 0.0, 0.0, 0.0 from text files
                subTableLongConst = cupy.concatenate((subTable, cupy.zeros((4, maxLength - subTable.shape[1] - 1))), axis=1)  # Adjust all trajectories to the same length by adding a table of NaN
                constantLengthTrajectoriesList.append(subTableLongConst)  

    shapeTrajectoriesNP(subTables, constantLengthTrajectoriesList)
    Trajs1Manip = cupy.array(constantLengthTrajectoriesList)  # Place arrays in columns in the form: nTrajs * nTypesData * nPointsOfTrajs
    # Trajs1Manip0 = Trajs1Manip[:, :, 0]  # Retrieve the first point of each trajectory
    Trajs1Manip = cupy.where(Trajs1Manip[:, :, 1:] == 0, cupy.nan, Trajs1Manip[:, :, 1:])  # Use where to replace values; [,:,1:] avoids the first point of each traj which can be the first zero of the time, later transformed into NaN
    # Trajs1Manip = numpy.concatenate((Trajs1Manip0, Trajs1Manip), axis=2)
    return Trajs1Manip, startingFrame

def openTrajectoryFileRetro(address):  # For backward compatibility: opening saved trajectory files in the format for Arrets_x_x.java, plugins for ImageJ
    file = open(address, "r")
    text = file.read()  
    text = re.sub(r"NumeroImageDebut ([0-9]+)", r"-1\t\1\t0.0\t0.0", text)  # \1 is a repetition of the part of the regex in parentheses; replaces string 'NumeroImageDebut' with -1 to indicate the starting image number following the format change to 4 columns
    text = re.sub("trajectoire\t[0-9]+\n", "", text)  # Remove line with string 'trajectoire' and the following number; two consecutive re.sub calls are faster than one combined
    text = re.sub(r'([\n\t][-]?[0-9]+)[\t]', r'\1.0\t', text)  # Converts integers of image numbers to floats; $ is the end of the string for a homogeneous float table
    text = '0.0' + text[1:-1]  # Corrects the absence of \n at the beginning to prevent detection of the first int by the regex above
    text = re.split('\n', text)
    # text = text.split('\n')  # Equivalent to re.split
    # text = text.splitlines()  # Equivalent to text.split()
    trajs2D = numpy.loadtxt(text, dtype=numpy.float64, delimiter='\t') 
    listMinusOne = (numpy.where(trajs2D == -1)[0] + 1).tolist()  # numpy.split requires a list, not np.array; offset by 1 because otherwise forces a split before the line starting with -1
    subTables = numpy.split(trajs2D, listMinusOne, axis=0) 
    subTablesForStartingFrame = subTables[0:-1]
    subTablesForStartingFrame.sort(key=lambda subTable: subTable[0][0])  # Sort by the smallest time; note the difference between sort and sorted
    startingFrame = subTablesForStartingFrame[0][0][1]  # OK    
    subTables.sort(key=lambda subTable: subTable.shape[0], reverse=True)  # Sort by decreasing length of trajectories
    maxLength = subTables[0].shape[0]
    constantLengthTrajectoriesList = []
    subTables[0] = numpy.transpose(subTables[0], (1, 0))
    subTables[0][0, :] = subTables[0][0, :] + subTables[0][1, -1]  # Convert to absolute time by adding the starting frame to the trajectory number
    subTables[0] = subTables[0][:, :-1]  # Remove the starting image number row
    constantLengthTrajectoriesList.append(subTables[0])

    # @njit
    def shapeTrajectoriesNP(subTables, constantLengthTrajectoriesList):
        for subTable in subTables[1:]:  
            if subTable.shape[0] > (3 * DURATION_MEASURE_SPEED + 2 * DELAY_CHECK_AROUND) * 4:  # Minimum duration threshold for stop checks
                subTable = numpy.transpose(subTable, (1, 0))  # Shape: nTypesData * nPointsOfTrajs         
                subTable[0, :] = subTable[0, :] + subTable[1, -1]  # Convert to absolute time
                subTable = subTable[:, :-1]  # Remove the starting image number row
                subTableLongConst = numpy.concatenate((subTable, numpy.zeros((4, maxLength - subTable.shape[1] - 1))), axis=1)  # Set all trajectories to the same length by adding a table of NaN
                constantLengthTrajectoriesList.append(subTableLongConst)                
        return constantLengthTrajectoriesList
    listeTrajLonguConstante = formeTrajsNP(subTables, listeTrajLonguConstante)
    Trajs1Manip = numpy.array(listeTrajLonguConstante)  # Place arrays in columns form: nTrajs*nTypesData*nPointsOfTrajs
    Trajs1Manip = numpy.where(Trajs1Manip == 0, numpy.nan, Trajs1Manip)     
    return Trajs1Manip, startFrame


############ Sort functions: ##########
    ########## 1) by areas ##########
    
def sortByAreas(Trajs1Manip):
    # Construct histogram of areas and Gaussian fit   
    Trajs1ManipTP = cupy.transpose(Trajs1Manip, (1, 0, 2))  # For area histogram: convert from nTrajs*nTypesData*nPointsOfTrajs to nTypesData*nTrajs*nPointsOfTrajs
    Trajs1ManipRS = cupy.reshape(Trajs1ManipTP, (Trajs1ManipTP.shape[0], -1), order="C")
    # @jit(parallel=True) #not effective on the first try
    def extractAreaHistogram(Trajs1ManipRS):
        Areas = Trajs1ManipRS[3].astype(cupy.float64)
        Areas = Areas[cupy.logical_not(cupy.isnan(Areas))]    # Remove NaN values
        histoAreas = cupy.histogram(Areas, bins=cupy.arange(1, cupy.sort(Areas)[len(Areas) - 1], PAS_HISTO_AIRES))        
        return histoAreas

    histoAreas = extractAreaHistogram(Trajs1ManipRS)
    binsHistoA = histoAreas[1][:len(histoAreas[1]) - 1]
    barresHistoA = histoAreas[0]
# print(barresHistoA)
    try:
        if GaussianSingleA == True: 
            popt, pcov = curve_fit(gaussian, binsHistoA.get(), barresHistoA.get(), p0=[1, 70, 20])  # p0 is the initial parameter estimation
            resFit = gaussian(binsHistoA, popt[0], popt[1], popt[2])  # NOTE: popt[2] = SD can be negative! # popt[0] is the Gaussian amplitude
            areaMin = popt[1] - 3 * abs(popt[2])
            areaMax = popt[1] + 3 * abs(popt[2])
        
        if TwoGaussiansA == True:     
            popt, pcov = curve_fit(mix2Gaussians, binsHistoA.get(), barresHistoA.get())
            resFit = mix2Gaussians(binsHistoA, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
            if (popt[0] > popt[3]):   # Sort by amplitude: choosing the Gaussian with the highest amplitude
               meanA = popt[1]
                SDA = abs(popt[2])  # NOTE: popt[2] = SD can be negative!
            else:
                meanA = popt[4]
                SDA = abs(popt[5])  # NOTE: popt[5] = SD can be negative!  
            areaMin = meanA - 3 * SDA
            areaMax = meanA + 3 * SDA
            
        if areaMin < 0 or areaMax > 1000:  # Handling outlier results
            areaMin = 10
            areaMax = 100
            resFit = gaussian(binsHistoA, numpy.argmax(histoAreas[0]), 45, 10) 
        
    except:  # In case of fitting failure
        areaMin = 10
        areaMax = 100
        resFit = gaussian(binsHistoA, numpy.argmax(histoAreas[0]), 45, 10)        

    Trajs1ManipForMean = cupy.copy(Trajs1Manip)  
    meanAreasTrajs1Manip = cupy.nanmean(Trajs1ManipForMean, axis=2)[:, 3]  # Calculate the mean areas     
    maskAreas = cupy.array((meanAreasTrajs1Manip > areaMin) & (meanAreasTrajs1Manip < areaMax))  # Create mask according to desired area conditions
    TrajsManipWithAreas = cupy.copy(Trajs1Manip)
    TrajsManipWithAreas = TrajsManipWithAreas[maskAreas]    
    return (TrajsManipWithAreas, areaMin, areaMax, binsHistoA, barresHistoA, resFit)


def manualAreaFilter(Trajs1Manip, areaMin, areaMax):
    Trajs1ManipForMean = cupy.copy(Trajs1Manip)  
    meanAreasTrajs1Manip = cupy.nanmean(Trajs1ManipForMean, axis=2)[:, 3]  # Calculate the mean areas 
    maskAreas = cupy.array((meanAreasTrajs1Manip > areaMin) & (meanAreasTrajs1Manip < areaMax))  # Create a mask according to the desired area conditions
    TrajsManipFilteredAreas = cupy.copy(Trajs1Manip)
    TrajsManipFilteredAreas = TrajsManipFilteredAreas[maskAreas]
    return TrajsManipFilteredAreas


########## 2) By speeds and distance calculation ##########

def detectSpeedThresholdsAndMeasureDistance(TrajsManipTrAires):  # Automatically launched method initially
    TrajsManipTrAiresSpeed = cupy.copy(TrajsManipTrAires) 
    # Construct speed histogram and Gaussian fit
    speedsX = (TrajsManipTrAiresSpeed[:, 1, :-DURATION_MEASURE_SPEED] - TrajsManipTrAiresSpeed[:, 1, DURATION_MEASURE_SPEED:]) / DURATION_MEASURE_SPEED  
    speedsX = cupy.reshape(speedsX, (speedsX.size), order="C")  # Convert to 1D
    speedsX = speedsX.astype(cupy.float64)
    speedsWithoutNan = speedsX[cupy.logical_not(cupy.isnan(speedsX))]  # Remove NaN values
    histoSpeed = cupy.histogram(speedsWithoutNan, bins=cupy.arange(0.1, cupy.sort(speedsWithoutNan)[len(speedsWithoutNan) - 1], HISTO_SPEED_STEP))        
    barsHistoV = histoSpeed[0]
    binsHistoV = histoSpeed[1][:len(histoSpeed[1]) - 1]  # Always one more bin limit in Numpy histograms       
    speedsMin = speedsWithoutNan[speedsWithoutNan > SPEED_MIN_FIT]  # Filter out low speeds for fitting
    histoSpeedsVMin = cupy.histogram(speedsMin, bins=cupy.arange(0.1, cupy.sort(speedsMin)[len(speedsMin) - 1], HISTO_SPEED_STEP))    
    barsHistoVMin = histoSpeedsVMin[0]
    binsHistoVMin = histoSpeedsVMin[1][:len(histoSpeedsVMin[1]) - 1]  # Always one more bin limit in Numpy histograms
    try: 
        popt, pcov = curve_fit(gaussian, binsHistoVMin.get(), barsHistoVMin.get(), p0=[500, 1, 0.5])  # Fit on values with minimum speed, p0 is the initial guess tuple
        resFit = gaussian(binsHistoV, popt[0], popt[1], popt[2])
        meanV = popt[1]
        SDV = abs(popt[2])  # NOTE: popt[2] = SD can be negative!  
        ############## With 2 Gaussians ###########################
        # popt, pcov = curve_fit(mix2Gaussians, binsHistoVMin.get(), barsHistoVMin.get())  # Fit on values with minimum speed
        # resFit = mix2Gaussians(binsHistoV, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        # if (popt[0] > popt[3]):  # Sort by amplitude: choosing the Gaussian with the highest amplitude
        #     meanV = popt[1]
        #     SDV = abs(popt[2])  # NOTE: popt[2] = SD can be negative!
        # else:
        #     meanV = popt[4]
        #     SDV = abs(popt[5])  # NOTE: popt[5] = SD can be negative!  
    except:  # Fitting failure
        meanV = 1  # Default values if fitting fails
        SDV = 1
        resFit = gaussian(binsHistoV, cupy.argmax(histoSpeed[0]), 0.5, 1)
    # Calculate the distance traveled
    speeds = cupy.column_stack((barsHistoV, binsHistoV))
    speeds = speeds.astype(float)
    maskV = cupy.array((speeds[:, 1] > meanV - 2 * SDV) & (speeds[:, 1] < meanV + 2 * SDV))
    sortedSpeedsV = speeds[maskV]
    distanceAfterSortingV = cupy.sum(sortedSpeedsV[:, 0] * sortedSpeedsV[:, 1])  # IMPORTANT: distance traveled in speed peak, in pixels
    return (meanV - 3 * SDV, meanV + 3 * SDV, distanceAfterSortingV, binsHistoV, barsHistoV, resFit)


def manualMeasureSpeed(TrajsManipTrAires):  # Recalculate speeds by calling the GUI for manual readjustment later 
    TrajsManipTrAiresSpeed = cupy.copy(TrajsManipTrAires) 
    # Construct speed histogram and Gaussian fit
    speedsX = (TrajsManipTrAiresSpeed[:, 1, :-DURATION_MEASURE_SPEED] - (TrajsManipTrAiresSpeed[:, 1, DURATION_MEASURE_SPEED:])) / DURATION_MEASURE_SPEED  
    speedsX = cupy.reshape(speedsX, (speedsX.size), order="C")  # Convert to 1D
    speedsX = speedsX.astype(float)
    speedsWithoutNan = speedsX[cupy.logical_not(cupy.isnan(speedsX))]  # Remove NaN values
    histoSpeed = cupy.histogram(speedsWithoutNan, bins=cupy.arange(0.1, cupy.sort(speedsWithoutNan)[len(speedsWithoutNan) - 1], HISTO_SPEED_STEP))    
    barsHistoV = histoSpeed[0]
    binsHistoV = histoSpeed[1][:len(histoSpeed[1]) - 1]  # Always one more bin limit in Numpy histograms
    speedsMin = speedsWithoutNan[speedsWithoutNan > SPEED_MIN_FIT]  # Filter out low speeds for fitting
    histoSpeedsVMin = cupy.histogram(speedsMin, bins=cupy.arange(0.2, cupy.sort(speedsMin)[len(speedsMin) - 1], HISTO_SPEED_STEP))    
    barsHistoVMin = histoSpeedsVMin[0]
    binsHistoVMin = histoSpeedsVMin[1][:len(histoSpeedsVMin[1]) - 1]  # Always one more bin limit in Numpy histograms
    try:
        popt, pcov = curve_fit(gaussian, binsHistoVMin.get(), barsHistoVMin.get(), p0=[40, 1, 0.5])  # Fit on values with minimum speed; p0 is the initial guess tuple
        resFit = gaussian(binsHistoV, popt[0], popt[1], popt[2])
        meanV = popt[1]
        SDV = abs(popt[2])  # NOTE: popt[2] = SD can be negative!    
        ############## With 2 Gaussians ###########################
        # popt, pcov = curve_fit(mix2Gaussians, binsHistoVMin.get(), barsHistoVMin.get())  # Fit on values with minimum speed
        # resFit = mix2Gaussians(binsHistoV, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        # if (popt[0] > popt[3]):  # Sort by amplitude: choosing the Gaussian with the highest amplitude
        #     meanV = popt[1]
        #     SDV = abs(popt[2])  # NOTE: popt[2] = SD can be negative!
        # else:
        #     meanV = popt[4]
        #     SDV = abs(popt[5])  # NOTE: popt[5] = SD can be negative!   
    except:  # Fitting failure
        meanV = 1  # Default values if fitting fails
        SDV = 1
        resFit = gaussian(binsHistoV, cupy.argmax(histoSpeed[0]), meanV, SDV)
    return (barsHistoV, binsHistoV, resFit)


def manualDistanceMeasure(barsHistoV, binsHistoV, vMin, vMax):  # Recalculate distances by calling the GUI after manual adjustment of speeds
    # Calculate the distance traveled
    speeds = cupy.column_stack((barsHistoV, binsHistoV))
    speeds = speeds.astype(float)
    maskV = cupy.array((speeds[:, 1] > vMin) & (speeds[:, 1] < vMax))
    filteredSpeedsV = speeds[maskV]
    distanceAfterFilteringV = cupy.sum(filteredSpeedsV[:, 0] * filteredSpeedsV[:, 1])  # IMPORTANT: distance traveled in speed peak, in pixels
    return distanceAfterFilteringV


############ Functions for quantifying stops: ##########
########## 1) Detection ##########

def detectAndSortStops(TrajsManipTrAires, areaMin, areaMax, vMin, vMax):
    # Detects the beginnings and ends of stops during trajectories
    TrajsManipStopsCP = cupy.asarray(TrajsManipTrAires)
    colCalculsV = cupy.copy(TrajsManipStopsCP[:, 1, :][:, cupy.newaxis, :])  # Copies for stop detection
    colCalculsV[colCalculsV != colCalculsV] = -100  # Trick to detect NaN
    # Stops for sorting
    TrajsManipStopsCP = cupy.concatenate((TrajsManipStopsCP, colCalculsV), axis=1)  # Add 5th column ([4])
    TrajsManipStopsCP = cupy.concatenate((TrajsManipStopsCP, colCalculsV), axis=1)  # Add 6th column ([5])
    condImmobile = (TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] - 
                    TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED + THRESHOLD_STOP_TEMP : - DELAY_CHECK_AROUND]) < THRESHOLD_STOP_X  # Boolean detecting positions below the stop speed threshold; True = immobile 
    TrajsManipStopsCP[:, 4, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] = condImmobile  # List of booleans indicating if stationary or mobile
    TrajsManipStopsCP[:, 4, :] = TrajsManipStopsCP[:, 4, :].astype(int)  # Convert booleans to 0 and 1 for subtraction by numpy.diff; astype.int does not work with NaN   
    condChangeStop = cupy.diff(TrajsManipStopsCP[:, 4, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND + 1])  # Detect transitions from stopped to mobile and from mobile to stopped; +1 needed for iteration over the axis of the table
    TrajsManipStopsCP[:, 5, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] = condChangeStop  # List of booleans indicating whether there is a transition from stopped to mobile, or from mobile to stopped, or no modification
    TrajsManipStopsCP[:, 5, :] = TrajsManipStopsCP[:, 5, :].astype(int)  # Convert booleans to integers (0 and 1) again to facilitate further calculations
    condStartChange = TrajsManipStopsCP[:, 5, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] == 1  # Transition from mobile to stopped; still mobile when this condition is true => one frame shift is necessary
    def applyStartCondition(b, condStartChange):  # WARNING: It is on column [4]
        condStartChange &= TrajsManipStopsCP[:, 4, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED + b : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND + b] == 1  # Remain stopped, verifying over multiple points
    for b in range(1, DURATION_STOP_CHECK):  
        applyStartCondition(b, condStartChange)  # Verify stop conditions over multiple points during the minimum stop duration
    condEndChange = TrajsManipStopsCP[:, 5, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED - 1 : -1 - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] == -1  # Transition from stopped to mobile; WARNING: - 1 shift is necessary because otherwise detected transitions from stopped to mobile at the end of trajectories (at the junction with analysis boundaries)
    # Conditions to be verified over multiple points in and around a stop (conditions for speeds before and after the stop, absolute area condition, absence of backward return)         
    # velocity conditions, area conditions ... must be initialized: done with the first shifted measurement    
    # Pre-stop velocity conditions essential to avoid stops in more or less stationary trajectories
    if vMin * COEFF_MinVMin < 0.1:  # Eliminate speeds that are too low for stop sorting
        vMinCorr = 0.1
    else: 
        vMinCorr = vMin * COEFF_MinVMin
    condSpeedsBeforeStop = ((TrajsManipArretsCP[:, 1, : - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_SPEED - DELAY_CHECK_AROUND] - TrajsManipArretsCP[:, 1, DELAY_CHECK_SPEED : - DURATION_MEASURE_SPEED - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND]) / DURATION_MEASURE_SPEED > vMinCorr) & ((TrajsManipArretsCP[:, 1, : - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_SPEED - DELAY_CHECK_AROUND] - TrajsManipArretsCP[:, 1, DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_AROUND]) / DURATION_MEASURE_SPEED < vMax * COEFF_MaxVIncrease)  # Initialization of speed condition object
    condAreas = (TrajsManipArretsCP[:, 3, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] > areaMin) & (TrajsManipArretsCP[:, 3, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] < areaMax)  # An absolute area condition, used for initializing the condition object
    condAbsReturnArr = (TrajsManipArretsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED - 1 : - 1 - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] - TrajsManipArretsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED + DURATION_MEASURE_RETURN - 1 : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND +    DURATION_MEASURE_RETURN - 1]) > THRESHOLD_RETURN_BACK  # The x position of the stop measured with delay must be less than or equal to the x position of the point measured at DURATION_MEASURE_RETURN before stopping, initializing the position condition object
    # print(TrajsManipArretsCP.shape)
    # print(condSpeedsBeforeStop.shape)
    # WARNING: No post-stop speed condition because immediate detachments followed by reattachments are ignored otherwise
    def applySpeedCondition(a, condSpeedsBeforeStop):
        condSpeedsBeforeStop &= ((TrajsManipStopsCP[:, 1, a: a - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_SPEED - DELAY_CHECK_AROUND] - TrajsManipStopsCP[:, 1, a + DELAY_CHECK_SPEED: a - DURATION_MEASURE_SPEED - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND]) / DURATION_MEASURE_SPEED > vMinCorr) & ((TrajsManipStopsCP[:, 1, a: a - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_SPEED - DELAY_CHECK_AROUND] - TrajsManipStopsCP[:, 1, a + DELAY_CHECK_SPEED: a - THRESHOLD_STOP_TEMP - DURATION_MEASURE_SPEED - DELAY_CHECK_AROUND]) / DURATION_MEASURE_SPEED < vMax * COEFF_MaxVIncrease)
        return condSpeedsBeforeStop
    def applyAreaCondition(a, condAreas):
        condAreas &= (TrajsManipStopsCP[:, 3, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED - a : - a - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] > areaMin) & (TrajsManipStopsCP[:, 3, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED - a : - a - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] < areaMax)  # An absolute area condition
        return condAreas
    def applyReturnCondition(a, condAbsReturnArr):  # , condSpeedsPostStop): # velocity conditions, area conditions ... must be initialized: done with the first shifted measurement       
        condAbsReturnArr &= (TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED - 1 + a: - 1 + a - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] - TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED + DURATION_MEASURE_RETURN - 1: - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND + DURATION_MEASURE_RETURN - 1]) > THRESHOLD_RETURN_BACK  # The x position of the stop measured with delay must be less than or equal to the x position of the point measured from 1 frame before stopping, then over DURATION_MEASURE_RETURN 
        return condAbsReturnArr   # , condSpeedsPostStop)
    # Application of area, speed, and backward return conditions over different intervals              
    for b in range(1, DURATION_CHECK_AROUND_SPEED):  # Measure speed before stopping from DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED by increments of 1, moving forward with value b; DURATION_CHECK_AROUND_SPEED must be less than or equal to DELAY_CHECK_SPEED to avoid artifact slowdown
        condsSpeed = applySpeedCondition(b, condSpeedsBeforeStop)
    for c in range(1, DURATION_CHECK_AROUND_AREA):  # Measure area immediately before stopping, moving back by value c in increments of 1 
        condsArea = applyAreaCondition(c, condAreas)  # , condSpeedsPostStop)  
    for d in range(1, DURATION_MEASURE_RETURN):  # Measure points distance before stopping from 1 frame before the stop to DURATION_MEASURE_RETURN - 1 after the start of the stop, moving forward with value d in increments of 1
        condsReturn = applyReturnCondition(d, condAbsReturnArr) 
    tupleConds = (condsSpeed, condsArea, condsReturn)         
    condSpeedsBeforeStop = tupleConds[0]
    condAreas = tupleConds[1]
    condAbsReturnArr = tupleConds[2]
    # condSpeedsPostStop = tupleConds[3]    
    # End of conditions on multiple points
    # Conditions for detecting the start and end of stops
    condStartStopInProgress = condStartChange & condSpeedsBeforeStop & condAreas & condAbsReturnArr
    condEndStopInProgress = condEndChange 
    condStopStartTraj = (TrajsManipStopsCP[:, 1, 0] - TrajsManipStopsCP[:, 1, THRESHOLD_STOP_TEMP]) < THRESHOLD_STOP_X  # Condition for detecting particles stopped at the start of the trajectory

    # Conditions for detecting stops outside sorting
    TrajsManipStopsCP = cupy.concatenate((TrajsManipStopsCP, colCalculsV), axis=1)  # Add 7th column ([6])
    TrajsManipStopsCP = cupy.concatenate((TrajsManipStopsCP, colCalculsV), axis=1)  # Add 8th column ([7])
    condImmobileOutsideSort = (TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] - TrajsManipStopsCP[:, 1, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED + THRESHOLD_STOP_TEMP_OUTSIDE_SORT : THRESHOLD_STOP_TEMP_OUTSIDE_SORT - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND]) < THRESHOLD_STOP_X  # Boolean detecting positions below stop speed threshold; True = immobile (?)
    TrajsManipStopsCP[:, 6, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] = condImmobileOutsideSort  # List of booleans indicating mobile or stopped
    TrajsManipStopsCP[:, 6, :] = TrajsManipStopsCP[:, 6, :].astype(int)  # Convert booleans to 0 and 1 for subtraction by numpy.diff
    condChangeStop = cupy.diff(TrajsManipStopsCP[:, 6, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND + 1])  # Detect transitions from stopped to mobile and from mobile to stopped; +1 needed for iteration over the axis of the table
    TrajsManipStopsCP[:, 7, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] = condChangeStop  # List of booleans indicating transitions from stop to mobile or mobile to stop, or no change
    TrajsManipStopsCP[:, 7, :] = TrajsManipStopsCP[:, 7, :].astype(int)  # Convert booleans to integers
    condStartChange1Point = TrajsManipStopsCP[:, 7, DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED : - THRESHOLD_STOP_TEMP - DELAY_CHECK_AROUND] == 1  # Transition from mobile to stop, unique to obtain stops that do not meet sorting conditions
    condStartStopWithoutConditions = condStartChange1Point
    # Retrieve the indices of the booleans
    posStartStops = numpy.asarray(condStartStopInProgress.get()).nonzero()  # Stops with sorting by conditions
    posEndStops = numpy.asarray(condEndStopInProgress.get()).nonzero()  
    posStopsStartTrajs = condStopStartTraj.nonzero()  # Indices of trajectories with stops at the start, in the form of a tuple of np arrays
    posStartsWithoutConditions = numpy.asarray(condStartStopWithoutConditions.get()).nonzero()  # All starts of stops for further connections (trajectories with stopped starts not included, retrieved in createStopList)
    # Sorting of stops without conditions: removal of starts of stops satisfying the conditions from the list of stops without conditions applied
    posStartStopsTr = numpy.concatenate((posStartStops[0], posStartStops[1]), axis=0)  # Concatenate the two lists
    posStartStopsTr = numpy.reshape(posStartStopsTr, (2, -1))  # Stack into 2 rows; -1 for automatic adjustment of the second dimension
    posStartStopsTr = posStartStopsTr.transpose()  # Transpose axes, reform pairs of initial data
    posStartsWithoutConditionsTr = numpy.concatenate((posStartsWithoutConditions[0], posStartsWithoutConditions[1]), axis=0)  # Concatenate the two lists
    posStartsWithoutConditionsTr = numpy.reshape(posStartsWithoutConditionsTr, (2, -1))  # Stack into 2 rows; -1 for automatic adjustment of the second dimension
    posStartsWithoutConditionsTr = posStartsWithoutConditionsTr.transpose()  # Transpose axes, maintain pairs of initial data; one line = 1 pair  
    posStartsWithoutConditionsTrTuple = [tuple(row) for row in posStartsWithoutConditionsTr]
    posStartStopsTrTuple = [tuple(row) for row in posStartStopsTr]
    indicesToKeep = [i for i, row in enumerate(posStartsWithoutConditionsTrTuple) if row not in posStartStopsTrTuple]
    posStartsWithoutConditionsTr = posStartsWithoutConditionsTr[indicesToKeep]  
    posStartsWithoutConditions = (posStartsWithoutConditionsTr[:, 0], posStartsWithoutConditionsTr[:, 1])
    # outputFile=open("C:\\Users\\Philippe\\Desktop\\testsPy\\tV.txt","w")
    # outputFile.write(textCheck)   
    return (posStartStops, posEndStops, posStopsStartTrajs, posStartsWithoutConditions)
########## 2): Connecting start and end and joining fragments of stopped trajectories ##########

def createStopList(TrajsManipStops, startFrame, posStartStops, posEndStops, posStopsStartTrajs, posStartsWithoutConditions):
    stopList = []  # Initial list from classic start stops (= transition from motion to stop then applying speed and area sorting)
    TrajsManipStops = TrajsManipStops.get()  # Convert back to numpy
    # Associate start stops with end stops
    for n in range(len(posStartStops[0])):  # posStartStops[0]= list of trajectory numbers, posStartStops[1]= list of positions of stops in their trajectory
        # Verify isolated bead (absence of contact with another bead at the stop moment): same time criteria and distance on x and y
        beadContactCond = (TrajsManipStops[posStartStops[0][n], 0, posStartStops[1][n]+ DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED] == TrajsManipStops[:, 0, :]) & (abs(TrajsManipStops[posStartStops[0][n], 1, posStartStops[1][n]+ DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED] - TrajsManipStops[:, 1, :]) < MIN_DISTANCE_TWO_BEADS_X) & (abs(TrajsManipStops[posStartStops[0][n], 2, posStartStops[1][n] + DURATION_MEASURE_SPEED + DELAY_CHECK_SPEED] - TrajsManipStops[:, 2, :]) < MIN_DISTANCE_TWO_BEADS_Y)
        if len(beadContactCond.nonzero()[0]) <= 1 :  # Case of no contact (trajectory meeting contact condition is the stop itself)
            duration = 0    
            endType = 0  # End stop type: 1 = Known duration, 0 = Unknown duration, -1 = Unknown duration joined, 2 known duration joined
            possibleEndNums = []           
            for m in range(len(posEndStops[0])):  # Find starts and ends belonging to the same stop
                if posStartStops[0][n] == posEndStops[0][m] and TrajsManipStops[posStartStops[0][n], 0, posStartStops[1][n]+DELAY_CHECK_AROUND+DURATION_MEASURE_SPEED+THRESHOLD_STOP_TEMP] <= TrajsManipStops[posEndStops[0][m], 0, posEndStops[1][m] + DELAY_CHECK_AROUND + DURATION_MEASURE_SPEED + THRESHOLD_STOP_TEMP] and abs(TrajsManipStops[posStartStops[0][n], 1, posStartStops[1][n] + DELAY_CHECK_AROUND + DURATION_MEASURE_SPEED + THRESHOLD_STOP_TEMP + 1] - TrajsManipStops[posEndStops[0][m], 1, posEndStops[1][m] + DELAY_CHECK_AROUND + DURATION_MEASURE_SPEED + THRESHOLD_STOP_TEMP + 1]) < MAX_STOP_DISPLACEMENT_X:
                    possibleEndNums.append(m)  # Indices of ends belonging to the same trajectory and in the correct position
            if len(possibleEndNums) > 0:  # Verify if detected end is the actual stop end (search for later detected ends and undetected later ends)
                maxEndNum = possibleEndNums[0]  # Search for the latest detected end
                for num in possibleEndNums:
                    if TrajsManipStops[posEndStops[0][num], 0, posEndStops[1][num]] > TrajsManipStops[posEndStops[0][maxEndNum], 0, posEndStops[1][maxEndNum]]:
                        maxEndNum = num    
                realTrajectory = numpy.extract(~numpy.isnan(TrajsManipStops[posStartStops[0][n], :, :]), TrajsManipStops[posStartStops[0][n], :, :])  # Find real points of the trajectory (i.e., non-NaN)
                realTrajectory = numpy.reshape(realTrajectory, (4, -1))  # Reshape to form 2D table with 4 columns for a single trajectory; dim -1 automatically calculates the length of the 2nd dimension (column length)
                if abs(trajReelle[1, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED] - trajReelle[1, -2]) < MAX_MOVE_THRESHOLD_STOP_X and abs(trajReelle[2, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED] - trajReelle[2, -2]) < MAX_MOVE_THRESHOLD_STOP_Y:
                    endType = 0  # Case where the actual stop is longer than the detected end(s): this becomes the unknown duration case
                else:  # Case where the actual end is among the detected end(s)
                    duration = TrajsManipStops[posEndStops[0][numMaxEnd], 0, posEndStops[1][numMaxEnd] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED] - TrajsManipStops[posStartStops[0][n], 0, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1]  # + STOP_THRESHOLD_TEMP
                    endType = 1  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType

            if endType == 0:  # Case of absence of end detection in the same trajectory: calculate known part of the unknown duration
                trajTime = TrajsManipStops[posStartStops[0][n], 0]
                realTrajTime = numpy.extract(~numpy.isnan(trajTime), trajTime)
                endTime = realTrajTime[realTrajTime.size - 1]  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType   
                duration = endTime - TrajsManipStops[posStartStops[0][n], 0, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED]  # Duration assumed when unknown is the end time of the trajectory minus the start time of the stop
            stopList.append([TrajsManipStops[posStartStops[0][n], 0, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + STOP_THRESHOLD_TEMP], TrajsManipStops[posStartStops[0][n], 1, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + STOP_THRESHOLD_TEMP],TrajsManipStops[posStartStops[0][n], 2, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + STOP_THRESHOLD_TEMP],TrajsManipStops[posStartStops[0][n], 3, posStartStops[1][n] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + STOP_THRESHOLD_TEMP],duration, posStartStops[0][n], endType])  # Still mobile when change condition is true => 1 frame shift necessary
    # 1) Detection and classification of completely stopped trajectories and 
    # detection and classification of trajectories with a stopped start and known duration stop
    print("List of stops before joining")
    print(len(stopList))
    fullStopList = []  # Completely stopped trajectories or incorrectly started to be joined with known or unknown ending stop
    startStopKnownEndTrajectoryList = []  # List of trajectories with stopped start or incorrectly started but with detected stop end to be joined with known or unknown ending stop
    startStopUnknownEndOutsideConditionsList = []  # Trajectories with stopped start outside sorting conditions, unknown end
    startStopKnownEndOutsideConditionsList = []  # Trajectories with stopped start outside sorting conditions, known end
    fullStopListCopy = [] 
    startStopKnownEndTrajectoryListCopy = [] 
    startStopUnknownEndOutsideConditionsListCopy = []  
    startStopKnownEndOutsideConditionsListCopy = []   
    if joinStops == True: 
        # 1) Classification of trajectories and stops 
        # 1-1) Classification of stopped start trajectories
        for pos in posStartsOfStops[0]:  # Reference starting at TrajsManipStops[:,:,0]
            # Form real trajectory = removing NaNs, allows reading end position of the trajectory
            realTrajectory = numpy.extract(~numpy.isnan(TrajsManipStops[pos.item(), :, :]), TrajsManipStops[pos.item(), :, :])
            realTrajectory = numpy.reshape(realTrajectory, (4, -1))  # Reshape to form a 2D table with 4 columns for a single trajectory; dim -1 automatically calculates the length of the second dimension (column length)        
        # 1-1-1) if that follows: search for condition of unchanged position throughout the stop: to be joined to the end of a stop 
        if abs(realTrajectory[1, 0] - realTrajectory[1, -2]) < MAX_MOVE_THRESHOLD_STOP_X and abs(realTrajectory[2, 0] - realTrajectory[2, -2]) < MAX_MOVE_THRESHOLD_STOP_Y and realTrajectory[0, 0] != startFrame: 
            # Verify unchanged spatial positions and trajectory appearing after the start of the film
            fullStopList.append((realTrajectory[0, 0], realTrajectory[1, 0], realTrajectory[2, 0], realTrajectory[0, -2], realTrajectory[1, -2], realTrajectory[2, -2], pos.item(), -1))  # Tuple to preserve time, X and Y (start), time, X and Y (end stop) and traj number           
        # 1-1-2) Search for known stop ends for condition or the stopped trajectory since its start corresponds to a classic stop end: to be joined to the end of a stop
        else:                 
            for z in range(len(posEndStops[0])):  # Reference starting at TrajsManipStops[:,:,DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED]
                if posEndStops[0][z] == pos.item():  # Confirm identity of trajectories (identity in list of end stops with identity in list of already stopped starts)
                    startStopKnownEndTrajectoryList.append((TrajsManipStops[posEndStops[0][z], 0, DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             TrajsManipStops[posEndStops[0][z], 1, DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             TrajsManipStops[posEndStops[0][z], 2, DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             TrajsManipStops[posEndStops[0][z], 0, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             TrajsManipStops[posEndStops[0][z], 1, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             TrajsManipStops[posEndStops[0][z], 2, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                             pos.item(), 2))  # Tuple to preserve time, X and Y (start), time, X, Y (end stop) and traj number
                      
        fullStopListCopy = fullStopList  # For conservation for display 
        startStopKnownEndTrajectoryListCopy = startStopKnownEndTrajectoryList
        # 1-2) Classification of stops with starts not meeting area, speed, etc. sorting conditions
        for numPos, pos in enumerate(posStartsWithoutConditions[0]):  # Reference starting at TrajsManipStops[:,:,DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1]
            realTrajectory = numpy.extract(~numpy.isnan(TrajsManipStops[pos.item(), :, :]), TrajsManipStops[pos.item(), :, :])
            realTrajectory = numpy.reshape(realTrajectory, (4, -1))  # Reshape to form a 2D table with 4 columns for a single trajectory; dim -1 automatically calculates the length of the second dimension (column length)
            startStop = posStartsWithoutConditions[1][numPos]  # In the reference frame of posStartsWithoutConditions    
        # 1-2-1) The following if statement: search for condition of unchanged position throughout the stop: to be joined to the end of a stop
            if abs(realTrajectory[1, startStop] - realTrajectory[1, -2]) < MAX_MOVE_THRESHOLD_STOP_X and abs(realTrajectory[2, startStop] - realTrajectory[2, -2]) < MAX_MOVE_THRESHOLD_STOP_Y and realTrajectory[0, 0] != startFrame:  # Verify unchanged spatial positions and trajectory appearing after the start of the film
                try:
                    startStopUnknownEndOutsideConditionsList.append((realTrajectory[0, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                     realTrajectory[1, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                     realTrajectory[2, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                     realTrajectory[0, -2], 
                                                                     realTrajectory[1, -2], 
                                                                     realTrajectory[2, -2], 
                                                                     pos.item(), -1))  # Tuple to preserve time, X and Y (start), time, X and Y (end stop), traj number, endType
                except:
                    startStopUnknownEndOutsideConditionsList.append((realTrajectory[0, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                     realTrajectory[1, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                     realTrajectory[2, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                     realTrajectory[0, -2], 
                                                                     realTrajectory[1, -2], 
                                                                     realTrajectory[2, -2], 
                                                                     pos.item(), -1))  # Tuple to preserve time, X and Y (start), time, X and Y (end stop), traj number, endType

            # 1-2-2) Search for known stop ends for condition or stopped trajectory since its start corresponds to a classic stop end: to be joined to the end of a stop
            else:                  
                for z in range(len(posEndStops[0])):  # Reference starting at TrajsManipStops[:,:,DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED]
                    if posEndStops[0][z] == pos.item():  # Verify identity of trajectories (identity in the list of stop ends with identity in the list of already stopped starts)
                        try:
                            startStopKnownEndOutsideConditionsList.append((realTrajectory[0, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                           realTrajectory[1, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                           realTrajectory[2, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1 + SHIFT_START_STOP_OUTSIDE_SORT], 
                                                                           TrajsManipStops[posEndStops[0][z], 0, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           TrajsManipStops[posEndStops[0][z], 1, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           TrajsManipStops[posEndStops[0][z], 2, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           pos.item(), 2))  # Tuple to preserve time, X and Y (start), time, X and Y (end stop), traj number, endType
                        except:
                            startStopKnownEndOutsideConditionsList.append((realTrajectory[0, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                           realTrajectory[1, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                           realTrajectory[2, startStop + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED + 1], 
                                                                           TrajsManipStops[posEndStops[0][z], 0, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           TrajsManipStops[posEndStops[0][z], 1, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           TrajsManipStops[posEndStops[0][z], 2, posEndStops[1][z] + DELAY_CHECK_SPEED + DURATION_MEASURE_SPEED], 
                                                                           pos.item(), 2))  # Tuple to preserve time, X and Y (start), time, X and Y (end stop), traj number, endType

        startStopUnknownEndOutsideConditionsListCopy = startStopUnknownEndOutsideConditionsList  # Necessary because startStopUnknownEndOutsideConditionsList is gradually emptied afterwards
        startStopKnownEndOutsideConditionsListCopy = startStopKnownEndOutsideConditionsList  # Necessary because startStopKnownEndOutsideConditionsList is gradually emptied afterwards
        # 2) Reconnecting interrupted stops with other stops
        # 2-1) Formation of groups to concatenate: a group attempted with a known start stop but unknown end
        groupedStopList = []  # List of tuples of stops at the same position        
        for stop in stopList:  # Search for joining solutions  # Case of interrupted stop: initiating the search  
            stopGroup = [stop]  # First member of the group is a stop with classic detection   # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType             
            for arr in fullStopList:  # Totally stopped trajectories   # FORMAT of stops: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                if stop[0] + stop[4] <= arr[0] and arr[0] - (stop[0] + stop[4]) < MAX_TEMP_JOIN_THRESHOLD and abs(stop[1] - arr[4]) < MAX_MOVE_THRESHOLD_STOP_X and abs(stop[2] - arr[5]) < MAX_MOVE_THRESHOLD_STOP_Y:  # Conditions: Trajectory completely stopped after a normal start stop and spatial positions unchanged
                    stopGroup.append(arr)
                    fullStopList.remove(arr) 
                    print("IC1")                       
            for arr in startStopKnownEndTrajectoryList:  # Trajectories with stopped start but detectable stop end  # FORMAT of stops: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                if stop[0] + stop[4] <= arr[0] and arr[0] - (stop[0] + stop[4]) < MAX_TEMP_JOIN_THRESHOLD and abs(stop[1] - arr[4]) < MAX_MOVE_THRESHOLD_STOP_X and abs(stop[2] - arr[5]) < MAX_MOVE_THRESHOLD_STOP_Y:  # Conditions: Stopped trajectory after normal start and spatial positions unchanged
                    stopGroup.append(arr)
                    startStopKnownEndTrajectoryList.remove(arr) 
                    print("IC2")     
            for arr in startStopUnknownEndOutsideConditionsList:  # Trajectories with stopped start outside sorting conditions, unknown end  # FORMAT of stops: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                if stop[0] + stop[4] <= arr[0] and arr[0] - (stop[0] + stop[4]) < MAX_TEMP_JOIN_THRESHOLD and abs(stop[1] - arr[4]) < MAX_MOVE_THRESHOLD_STOP_X and abs(stop[2] - arr[5]) < MAX_MOVE_THRESHOLD_STOP_Y:  # Conditions: Trajectory completely stopped after a normal start stop and spatial positions unchanged
                    stopGroup.append(arr)
                    startStopUnknownEndOutsideConditionsList.remove(arr)   
                    print("DNT1") 
            for arr in startStopKnownEndOutsideConditionsList:  # Trajectories with stopped start outside sorting conditions, known end  # FORMAT of stops: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                if stop[0] + stop[4] <= arr[0] and arr[0] - (stop[0] + stop[4]) < MAX_TEMP_JOIN_THRESHOLD and abs(stop[1] - arr[4]) < MAX_MOVE_THRESHOLD_STOP_X and abs(stop[2] - arr[5]) < MAX_MOVE_THRESHOLD_STOP_Y:  # Conditions: Trajectory completely stopped after a normal start stop and spatial positions unchanged
                    stopGroup.append(arr)
                    startStopKnownEndOutsideConditionsList.remove(arr)  
                    print("DNT2")                        
            if len(stopGroup) > 1:  # Case grouping for concatenation has worked
                groupedStopList.append(stopGroup)
        print("Stops to be joined")
        print(len(groupedStopList))
        # 2-2) Concatenation of stops in each group 
        modificationCount = 0
        for group in groupedStopList:  # group[0] = stop with classic start            
               maxStopTime = group[0]
               maxTime = maxStopTime[0]   
               endType = maxStopTime[6]  # End type of the initially detected stop
               for n, stop in enumerate(group):  # Search for the latest start stop
                   if n > 0:  # Eviction of the first stop 
                       if stop[0] > maxTime:  # FORMAT of stops in trajectories with stopped starts: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                           maxStopTime = stop 
                           maxTime = stop[0]  # Could be stop[3]
                           endType = stop[7]  # End type of the entirely stopped trajectory
               for stopToModify in stopList:  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
                   if stopToModify == group[0]:  # Looking for original incomplete stop  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
                       stopToModify[4] = maxStopTime[3] - stopToModify[0] + STOP_THRESHOLD_TEMP  # Duration: end time of the group's max stop minus the start time of the stop to modify
                       stopToModify[6] = endType
                       modificationCount += 1
        print("Modified stops")
        print(modificationCount)           
    return (stopList, fullStopListCopy, startStopKnownEndTrajectoryListCopy, startStopUnknownEndOutsideConditionsListCopy, startStopKnownEndOutsideConditionsListCopy)  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
############################################################### GUI Functions ##################################################
###################################################################################################################################

######################### Plot window for areas, speeds, trajectories with stops ###################################
def reshapePlotUniqueTrajectory(TrajsManipAreas):  # Creates a unique trajectory for faster plotting in pyqtgraph
    shapeTrajs = numpy.array(TrajsManipAreas.shape)
    reshapedTrajsForPlot = numpy.copy(TrajsManipAreas[:, numpy.array([0, 1, 3]), :])
    reshapedTrajsForPlot = numpy.transpose(reshapedTrajsForPlot, (1, 0, 2))
    reshapedTrajsForPlot = numpy.reshape(reshapedTrajsForPlot, (3, shapeTrajs[0] * shapeTrajs[2])) 
    return reshapedTrajsForPlot


class PlotTrajectories(QDialog): 
    global maxLength, displayAllStops
    returnValue = pyqtSignal(int)
    def __init__(self, address, trajsNP, startFrame, reshapedTrajsForPlot, allStopsTuple, minArea, maxArea, binsHistoA, barsHistoA, fitResultsA, minSpeed, maxSpeed, distanceAfterSortingV, binsHistoV, barsHistoV, fitResultsV):
        super().__init__()
        self.window = QtWidgets.QWidget()       
        self.window.setStyleSheet('background-color: black')
        self.address = address    
        self.splitAddress = os.path.split(self.address)  # For saving address composition  # Separates the folder path from the file name  
        self.trajsNP = trajsNP
        self.startFrame = startFrame
        self.reshapedTrajsForPlot = reshapedTrajsForPlot
        self.allStopsTuple = allStopsTuple
        self.stops = allStopsTuple[0]
        self.minArea = minArea
        self.maxArea = maxArea
        self.binsHistoA = binsHistoA
        self.barsHistoA = barsHistoA
        self.fitResultsA = fitResultsA
        self.distanceAfterSortingV = distanceAfterSortingV
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.binsHistoV = binsHistoV
        self.barsHistoV = barsHistoV
        self.fitResultsV = fitResultsV
        self.window.setWindowTitle(self.address)
        
    def plotGraph(self):
        global guiBlocked 
        # Histogram of areas
        self.areaHistogram = pyqtgraph.PlotWidget(self.window) 
        self.areaHistogram.plot(self.binsHistoA.get(), self.barsHistoA.get())
        self.areaHistogram.plot(self.binsHistoA.get(), self.fitResultsA.get(), pen='r')
        self.areaSelection = pyqtgraph.LinearRegionItem(values=(self.minArea, self.maxArea), orientation="vertical", brush=(255, 0, 0, 50))
        self.areaHistogram.addItem(self.areaSelection)
        self.areaHistogram.setGeometry(500, 10, 700, 300)
        # Histogram of speeds
        self.speedHistogram = pyqtgraph.PlotWidget(self.window)       
        self.speedHistogram.plot(self.binsHistoV.get(), self.barsHistoV.get())
        self.speedHistogram.plot(self.binsHistoV.get(), self.fitResultsV.get(), pen="g")
        self.speedSelection = pyqtgraph.LinearRegionItem(values=(self.minSpeed, self.maxSpeed), orientation="vertical", brush=(0, 255, 0, 50))
        self.speedHistogram.addItem(self.speedSelection)  
        self.speedHistogram.setGeometry(1210, 10, 700, 300)
        # Plot of trajectories
        self.trajectoryGraph = pyqtgraph.PlotWidget(self.window) 
        self.trajectoryGraph.disableAutoRange()  # Significantly speeds up trajectory display 
        self.trajectoryGraph.setXRange(self.reshapedTrajsForPlot[0][1].get(), self.reshapedTrajsForPlot[0][1].get() + 5000)  # Shift the graph to the start of the trajectories in absolute time; 5000 frames is 100 seconds at 50 fps
        self.trajectoryGraph.setYRange(0, 2048)  # Sensor size is 2048 pixels
        self.trajectoryGraphPlotItem = pyqtgraph.PlotDataItem(self.reshapedTrajsForPlot[0].get(), self.reshapedTrajsForPlot[1].get(), connect='finite')  # Reshaping into one table speeds up the display
        self.trajectoryGraph.addItem(self.trajectoryGraphPlotItem)
        self.trajectoryGraph.setGeometry(0, 320, 2000, 700)
         # Managing stops for the trajectory graph
        self.knownDurationStops = []  # Known durations
        self.unknownDurationStops = []  # Unknown durations
        self.joinedKnownDurationStops = []  # Known durations, joined
        self.joinedUnknownDurationStops = []  # Unknown durations, joined
        for stop in self.stops:  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
            if stop[6] == 1:  # Case of known duration
                self.knownDurationStops.append(stop)
            if stop[6] == 0:  # Case of unknown duration
                self.unknownDurationStops.append(stop)
            if stop[6] == 2:  # Case of known duration but joined
                self.joinedKnownDurationStops.append(stop)
            if stop[6] == -1:  # Case of unknown duration but joined
                self.joinedUnknownDurationStops.append(stop)
        self.knownDurationStopsNP = numpy.transpose(numpy.array(self.knownDurationStops))  # For time and columns accessible as self.stopsNP[0], self.stopsNP[1], ...
        self.unknownDurationStopsNP = numpy.transpose(numpy.array(self.unknownDurationStops))
        self.joinedKnownDurationStopsNP = numpy.transpose(numpy.array(self.joinedKnownDurationStops))  
        self.joinedUnknownDurationStopsNP = numpy.transpose(numpy.array(self.joinedUnknownDurationStops))      
        try :
            endTimeKnownStops = self.knownDurationStopsNP[0] + self.knownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeKnownStops = numpy.insert(self.knownDurationStopsNP[0], numpy.arange(1, len(self.knownDurationStopsNP[0]) + 1), endTimeKnownStops)
            couplesXKnownStops = numpy.insert(self.knownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.knownDurationStopsNP[1]) + 1), self.knownDurationStopsNP[1] - 0.5)
            self.graphKnownStops = pyqtgraph.PlotDataItem(couplesTimeKnownStops, couplesXKnownStops, pen='g', connect='pairs')           
            self.startKnownDurationStopsGraph = pyqtgraph.ScatterPlotItem(self.knownDurationStopsNP[0], self.knownDurationStopsNP[1], pen='g', brush='g')
            self.trajectoryGraph.addItem(self.graphKnownDurationStops)   # Duration of stops
            self.trajectoryGraph.addItem(self.startKnownDurationStopsGraph)  # Start points of stops
        except:  # For empty list of known duration stops
             pass  
        try:
            endTimesUnknownDurationStops = self.unknownDurationStopsNP[0] + self.unknownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeUnknownDurationStops = numpy.insert(self.unknownDurationStopsNP[0], numpy.arange(1, len(self.unknownDurationStopsNP[0]) + 1), endTimesUnknownDurationStops)
            couplesXUnknownDurationStops = numpy.insert(self.unknownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.unknownDurationStopsNP[1]) + 1), self.unknownDurationStopsNP[1] - 0.5)
            self.graphUnknownDurationStops = pyqtgraph.PlotDataItem(couplesTimeUnknownDurationStops, couplesXUnknownDurationStops, pen='r', connect='pairs')
            self.startUnknownDurationStopsGraph = pyqtgraph.ScatterPlotItem(self.unknownDurationStopsNP[0], self.unknownDurationStopsNP[1], pen='r', brush='r')
            self.trajectoryGraph.addItem(self.graphUnknownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startUnknownDurationStopsGraph)  # Start points of stops
        except:  # For empty list of unknown duration stops
            pass
        try:
            endTimesJoinedKnownDurationStops = self.joinedKnownDurationStopsNP[0] + self.joinedKnownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeJoinedKnownDurationStops = numpy.insert(self.joinedKnownDurationStopsNP[0], numpy.arange(1, len(self.joinedKnownDurationStopsNP[0]) + 1), endTimesJoinedKnownDurationStops)
            couplesXJoinedKnownDurationStops = numpy.insert(self.joinedKnownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.joinedKnownDurationStopsNP[1]) + 1), self.joinedKnownDurationStopsNP[1] - 0.5)
            self.graphJoinedKnownDurationStops = pyqtgraph.PlotDataItem(couplesTimeJoinedKnownDurationStops, couplesXJoinedKnownDurationStops, pen='c', connect='pairs')    
            self.startJoinedKnownDurationStopsGraph = pyqtgraph.ScatterPlotItem(self.joinedKnownDurationStopsNP[0], self.joinedKnownDurationStopsNP[1], pen='c', brush='c')
            self.trajectoryGraph.addItem(self.graphJoinedKnownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startJoinedKnownDurationStopsGraph)  # Start points of stops
        except:  # For empty list of joined known duration stops
            pass
        try:
            endTimesJoinedUnknownDurationStops = self.joinedUnknownDurationStopsNP[0] + self.joinedUnknownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeJoinedUnknownDurationStops = numpy.insert(self.joinedUnknownDurationStopsNP[0], numpy.arange(1, len(self.joinedUnknownDurationStopsNP[0]) + 1), endTimesJoinedUnknownDurationStops)
            couplesXJoinedUnknownDurationStops = numpy.insert(self.joinedUnknownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.joinedUnknownDurationStopsNP[1]) + 1), self.joinedUnknownDurationStopsNP[1] - 0.5)
            self.graphJoinedUnknownDurationStops = pyqtgraph.PlotDataItem(couplesTimeJoinedUnknownDurationStops, couplesXJoinedUnknownDurationStops, pen='y', connect='pairs')    
            self.startJoinedUnknownDurationStopsGraph = pyqtgraph.ScatterPlotItem(self.joinedUnknownDurationStopsNP[0], self.joinedUnknownDurationStopsNP[1], pen='y', brush='y')
            self.trajectoryGraph.addItem(self.graphJoinedUnknownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startJoinedUnknownDurationStopsGraph)  # Start points of stops
        except:  # For empty list of joined unknown duration stops
            pass
        if displayAllStops == True:
            self.startTimesOutsideConditions = []  # Known or unknown ends
            self.startPositionsOutsideConditions = []  # Known or unknown ends
            try :  # tupleAllStops = (stopList, fullStopList, startStopKnownEndList, startStopUnknownEndOutsideConditionsList, startStopKnownEndOutsideConditionsList)
            # FORMAT of items in other lists: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
                for arr in self.allStopsTuple[3]:  # startStopUnknownEndOutsideConditionsList
                    self.startTimesOutsideConditions.append(arr[0])
                    self.startPositionsOutsideConditions.append(arr[1])
                for arr in self.allStopsTuple[4]:  # startStopKnownEndOutsideConditionsList
                    self.startTimesOutsideConditions.append(arr[0])
                    self.startPositionsOutsideConditions.append(arr[1])
                    self.graphStartStopsOutsideConditions = pyqtgraph.ScatterPlotItem(self.startTimesOutsideConditions, self.startPositionsOutsideConditions, symbol='x', pen='y')        
                    self.trajectoryGraph.addItem(self.graphStartStopsOutsideConditions)  # Start points of stops
            except:  # For empty outside condition stops list
                pass   
            self.startTimesStopped = []  # Known or unknown ends
            self.startPositionsStopped = []  # Known or unknown ends
            try:  # tupleAllStops = (stopList, fullStopList, startStopKnownEndList, startStopUnknownEndOutsideConditionsList, startStopKnownEndOutsideConditionsList)
            # FORMAT of items in other lists: tuple time, X and Y (start), time, X, Y (end stop), traj number, endType
               for arr in self.allStopsTuple[1]:  # startStopKnownEndList
                   self.startTimesStopped.append(arr[0])
                   self.startPositionsStopped.append(arr[1])
               for arr in self.allStopsTuple[2]:  # startStopUnknownEndList
                   self.startTimesStopped.append(arr[0])
                   self.startPositionsStopped.append(arr[1])
               self.graphStoppedTrajectories = pyqtgraph.ScatterPlotItem(self.startTimesStopped, self.startPositionsStopped, symbol='x', pen='r')        
               self.trajectoryGraph.addItem(self.graphStoppedTrajectories)  # Start points of stops
           except:  # For empty known duration stops list
               pass
       self.trajectoryGraph.scene().sigMouseClicked.connect(self.click)  # Mouse click connection
       buttonValidate = QPushButton(self.window)
       buttonValidate.setText("Validate")       
       buttonValidate.move(10, 10)
       buttonValidate.resize(480, 55)
       buttonValidate.setStyleSheet("background-color: lime")
       buttonValidate.setFont(QFont('Arial', 17))
       buttonValidate.clicked.connect(self.saveAndClose)
       buttonRecalculate = QPushButton(self.window)
       buttonRecalculate.setText("Recalculate")
       buttonRecalculate.move(10, 65)
       buttonRecalculate.resize(480, 55)
       buttonRecalculate.setFont(QFont('Arial', 17))
       buttonRecalculate.setStyleSheet("background-color: lightgrey")
       buttonRecalculate.clicked.connect(self.manualRecalculate)
       buttonBack = QPushButton(self.window)
       buttonBack.setText("Back")
       buttonBack.move(10, 120)
       buttonBack.resize(480, 55)
       buttonBack.setFont(QFont('Arial', 17))
       buttonBack.setStyleSheet("background-color: lightgrey")
       buttonBack.clicked.connect(self.requestPrevious)
       buttonIgnore = QPushButton(self.window)
       buttonIgnore.setText("Ignore")
       buttonIgnore.move(10, 175)     
       buttonIgnore.resize(480, 55)
       buttonIgnore.setFont(QFont('Arial', 17))
       buttonIgnore.setStyleSheet("background-color: orangered")
       buttonIgnore.clicked.connect(self.ignoreResults)
       buttonOpenFile = QPushButton("Video Processing", self.window)
       buttonOpenFile.clicked.connect(self.createVideoVerifications)   
       buttonOpenFile.move(10, 230)
       buttonOpenFile.resize(480, 55)
       buttonOpenFile.setFont(QFont('Arial', 17))
       buttonOpenFile.setStyleSheet("background-color: lightgrey")       
       self.knownDurationStopsNPInitial = self.knownDurationStopsNP
       self.unknownDurationStopsNPInitial = self.unknownDurationStopsNP
       self.joinedKnownDurationStopsNPInitial = self.joinedKnownDurationStopsNP
       self.joinedUnknownDurationStopsNPInitial = self.joinedUnknownDurationStopsNP           
       if guiBlocked == False:
           self.window.showMaximized()
           else: 
               self.window.showMaximized()
               self.saveAndClose()  # Save function shared for manual analysis via plotTrajs
               self.window.close()
# self.window.setWindowFlags(Qt.WindowStaysOnTopHint)
    def click(self, event): 
        pointerPos = self.trajectoryGraph.getPlotItem().vb.mapSceneToView(event._scenePos)        
        # print(self.unknownDurationStopsNP) # time, X, Y
        # print(pointerPos.x(), pointerPos.y())   
        if event.button() == QtCore.Qt.LeftButton:  # Remove stop
            try:  # For empty list of known duration stops
                proximityConditionC = ((abs(self.knownDurationStopsNP[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & 
                                       (abs(self.knownDurationStopsNP[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X))  
                if numpy.count_nonzero(proximityConditionC) > 0:  # Count the number of True instances of the tested condition
                    positionsStopRemovedC = numpy.array(proximityConditionC)
                    positionsStopKeptC = ~positionsStopRemovedC
                    self.knownDurationStopsNP = self.knownDurationStopsNP[:, positionsStopKeptC]
                    self.graphKnownDurationStops.clear()
                    self.graphKnownDurationStops.addPoints(self.knownDurationStopsNP[0], self.knownDurationStopsNP[1], pen='g')
            except:  # For empty list of known duration stops
                pass

            try:  # For empty list of unknown duration stops          
                proximityConditionI = ((abs(self.unknownDurationStopsNP[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & 
                                       (abs(self.unknownDurationStopsNP[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X))  
                if numpy.count_nonzero(proximityConditionI) > 0:
                    positionsStopRemovedI = numpy.array(proximityConditionI)
                    positionsStopKeptI = ~positionsStopRemovedI
                    self.unknownDurationStopsNP = self.unknownDurationStopsNP[:, positionsStopKeptI]
                    self.graphUnknownDurationStops.clear()
                    self.graphUnknownDurationStops.addPoints(self.unknownDurationStopsNP[0], self.unknownDurationStopsNP[1], pen='r')
            except:  # For empty list of unknown duration stops
                pass

            try:
                proximityConditionCR = ((abs(self.joinedKnownDurationStopsNP[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & 
                                        (abs(self.joinedKnownDurationStopsNP[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X))  
                if numpy.count_nonzero(proximityConditionCR) > 0:
                    positionsStopRemovedCR = numpy.array(proximityConditionCR)
                    positionsStopKeptCR = ~positionsStopRemovedCR
                    self.joinedKnownDurationStopsNP = self.joinedKnownDurationStopsNP[:, positionsStopKeptCR]
                    self.graphJoinedKnownDurationStops.clear()
                    self.graphJoinedKnownDurationStops.addPoints(self.joinedKnownDurationStopsNP[0], self.joinedKnownDurationStopsNP[1], pen='c')
            except:  # For empty list of joined known duration stops
                pass

            try:
                proximityConditionIR = ((abs(self.joinedUnknownDurationStopsNP[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & 
                                        (abs(self.joinedUnknownDurationStopsNP[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X))  
                if numpy.count_nonzero(proximityConditionIR) > 0:
                    positionsStopRemovedIR = numpy.array(proximityConditionIR)
                    positionsStopKeptIR = ~positionsStopRemovedIR
                    self.joinedUnknownDurationStopsNP = self.joinedUnknownDurationStopsNP[:, positionsStopKeptIR]
                    self.graphJoinedUnknownDurationStops.clear()
                    self.graphJoinedUnknownDurationStops.addPoints(self.joinedUnknownDurationStopsNP[0], self.joinedUnknownDurationStopsNP[1], pen='b')
            except:  # For empty list of joined unknown duration stops
                pass
        if event.button() == QtCore.Qt.RightButton:  # Replace stop  
            try:
                proximityConditionCInit = (abs(self.knownDurationStopsNPInitial[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & (abs(self.knownDurationStopsNPInitial[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X) 
                if numpy.count_nonzero(proximityConditionCInit) > 0:
                    positionsToReplaceC = numpy.array(proximityConditionCInit)
                    stopToReplaceC = self.knownDurationStopsNPInitial[:, positionsToReplaceC]
                    self.knownDurationStopsNP = numpy.concatenate((self.knownDurationStopsNP, stopToReplaceC), axis=1)
                    self.graphKnownDurationStops.clear()
                    self.graphKnownDurationStops.addPoints(self.knownDurationStopsNP[0], self.knownDurationStopsNP[1], pen='g')
            except:  # For empty list of known duration stops
                pass
            try:
                proximityConditionIInit = (abs(self.unknownDurationStopsNPInitial[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & (abs(self.unknownDurationStopsNPInitial[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X) 
                if numpy.count_nonzero(proximityConditionIInit) > 0:
                    positionsToReplaceI = numpy.array(proximityConditionIInit)
                    stopToReplaceI = self.unknownDurationStopsNPInitial[:, positionsToReplaceI]
                    self.unknownDurationStopsNP = numpy.concatenate((self.unknownDurationStopsNP, stopToReplaceI), axis=1)
                    self.graphUnknownDurationStops.clear()
                    self.graphUnknownDurationStops.addPoints(self.unknownDurationStopsNP[0], self.unknownDurationStopsNP[1], pen='r')
            except:  # For empty list of unknown duration stops
                pass
            try:            
                proximityConditionCRInit = (abs(self.joinedKnownDurationStopsNPInitial[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & (abs(self.joinedKnownDurationStopsNPInitial[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X) 
                if numpy.count_nonzero(proximityConditionCRInit) > 0:
                    positionsToReplaceCR = numpy.array(proximityConditionCRInit)
                    stopToReplaceCR = self.joinedKnownDurationStopsNPInitial[:, positionsToReplaceCR]
                    self.joinedKnownDurationStopsNP = numpy.concatenate((self.joinedKnownDurationStopsNP, stopToReplaceCR), axis=1)
                    self.graphJoinedKnownDurationStops.clear()
                    self.graphJoinedKnownDurationStops.addPoints(self.joinedKnownDurationStopsNP[0], self.joinedKnownDurationStopsNP[1], pen='c')
            except:  # For empty list of joined known duration stops
                pass   
            try:            
                proximityConditionIRInit = (abs(self.joinedUnknownDurationStopsNPInitial[0, :] - pointerPos.x()) < CLICK_PROXIMITY_THRESHOLD_T) & (abs(self.joinedUnknownDurationStopsNPInitial[1, :] - pointerPos.y()) < CLICK_PROXIMITY_THRESHOLD_X) 
                if numpy.count_nonzero(proximityConditionIRInit) > 0:
                    positionsToReplaceIR = numpy.array(proximityConditionIRInit)
                    stopToReplaceIR = self.joinedUnknownDurationStopsNPInitial[:, positionsToReplaceIR]
                    self.joinedUnknownDurationStopsNP = numpy.concatenate((self.joinedUnknownDurationStopsNP, stopToReplaceIR), axis=1)
                    self.graphJoinedUnknownDurationStops.clear()
                    self.graphJoinedUnknownDurationStops.addPoints(self.joinedUnknownDurationStopsNP[0], self.joinedUnknownDurationStopsNP[1], pen='b')
            except:  # For empty list of joined unknown duration stops
                pass
    def saveAndClose(self):  # Save stop results after manual verification AND during autonomous use
        print("Saving stops and graphs")
        outputAddressCalc = self.splitAddress[0] + "Calculations.txt"
        outputText = "\n"
        frequency = len(self.stops) / self.distanceAfterSortingV * PIX_MICRON_CONVERSION_FACTOR 
        stopsText = ""
        knownDurationCount = 0
        for stop in self.stops:  # FORMAT of items in self.stops: absoluteTime, X, Y, area, duration, traj number, endType
            if stop[6] > 0:  # Case of known duration # 1: originally known duration; 2: known duration after joining     
                knownDurationCount += 1
                stopsText += str(stop[4] * TIME_CONVERSION_FACTOR) + "\n"
            else:
                stopsText += str(20000) + '\n'  
        outputText += str(frequency) + '\n' + str(len(self.stops)) + '\n' + str(knownDurationCount) + '\n' + str(len(self.stops) - knownDurationCount) + '\n' + str(self.distanceAfterSortingV * PIX_MICRON_CONVERSION_FACTOR) + '\n' + str(self.minSpeed * PIX_MICRON_CONVERSION_FACTOR / TIME_CONVERSION_FACTOR) + '\n' + str((self.minSpeed + self.maxSpeed) / 2 * PIX_MICRON_CONVERSION_FACTOR / TIME_CONVERSION_FACTOR) + '\n' + str(self.maxSpeed * PIX_MICRON_CONVERSION_FACTOR / TIME_CONVERSION_FACTOR) + '\n' + str(self.minArea) + '\n' + str(self.maxArea) + '\n\n'
        outputText += stopsText
        outputFileCalc = open(outputAddressCalc, "w")
        outputFileCalc.write(outputText)
        self.returnValue.emit(1)  # Not used autonomously  
        outputAddressTrajs = self.splitAddress[0] + "Trajectories.png"
        exportTrajs = pyqtgraph.exporters.ImageExporter(self.trajectoryGraph.getPlotItem())  # Export works on the plotItem of a PlotWidget
        exportTrajs.export(outputAddressTrajs)
        self.trajectoryGraph.close()       
        outputAddressHistoA = self.splitAddress[0] + "Areas.png"
        exportAreas = pyqtgraph.exporters.ImageExporter(self.areaHistogram.getPlotItem())  # Export works on the plotItem of a PlotWidget
        exportAreas.export(outputAddressHistoA)
        self.areaHistogram.close()
        outputAddressHistoV = self.splitAddress[0] + "Speeds.png"
        exportSpeeds = pyqtgraph.exporters.ImageExporter(self.speedHistogram.getPlotItem())  # Export works on the plotItem of a PlotWidget
       exportSpeeds.export(outputAddressHistoV)
       self.speedHistogram.close()        
       self.window.close()
       return
   
    def manualRecalculate(self): 
        areas = self.areaSelection.getRegion()
        self.minArea = areas[0]
        self.maxArea = areas[1]        
        manipulatedTrajectoriesAreas = sortByManualAreas(self.trajsNP, self.minArea, self.maxArea)
        self.barsHistoV, self.binsHistoV, self.fitResultsV = measureManualSpeed(manipulatedTrajectoriesAreas)    
        speeds = self.speedSelection.getRegion()
        self.minSpeed = speeds[0]
        self.maxSpeed = speeds[1]
        self.distanceAfterSortingV = measureManualDistance(self.barsHistoV, self.binsHistoV, self.minSpeed, self.maxSpeed)   
        self.speedHistogram.clear()
        self.speedHistogram.plot(self.binsHistoV.get(), self.barsHistoV.get())
        self.speedHistogram.plot(self.binsHistoV.get(), self.fitResultsV.get(), pen="g")
        self.speedSelection = pyqtgraph.LinearRegionItem(values=(self.minSpeed, self.maxSpeed), orientation="vertical", brush=(0, 255, 0, 50))
        self.speedHistogram.addItem(self.speedSelection)      
        stopPositions = detectSortedStops(manipulatedTrajectoriesAreas, self.minArea, self.maxArea, self.minSpeed, self.maxSpeed)
        self.allStopsTuple = createStopList(manipulatedTrajectoriesAreas, self.startFrame, stopPositions[0], stopPositions[1], stopPositions[2], stopPositions[3])  # Arguments: manipulatedTrajectories, start frame number, start stop positions, end stop positions, positions without conditions
        self.stops = self.allStopsTuple[0]
        self.reshapedTrajsForPlot = reshapePlotUniqueTrajectory(manipulatedTrajectoriesAreas)     
        self.trajectoryGraph.clear()
        self.trajectoryGraphPlotItem = pyqtgraph.PlotDataItem(self.reshapedTrajsForPlot[0].get(), self.reshapedTrajsForPlot[1].get(), connect='finite')  # Reshaping into one table speeds up the display 
        self.trajectoryGraph.addItem(self.trajectoryGraphPlotItem)
        # Managing stops for the trajectory graph
        self.knownDurationStops = []  # Known durations
        self.unknownDurationStops = []  # Unknown durations
        self.joinedKnownDurationStops = []  # Known durations, joined
        self.joinedUnknownDurationStops = []  # Unknown durations, joined    
        for stop in self.stops:  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
            if stop[6] == 1:  # Case of known duration
                self.knownDurationStops.append(stop)
            if stop[6] == 0:  # Case of unknown duration
                self.unknownDurationStops.append(stop)
            if stop[6] == 2:  # Case of known duration but joined
                self.joinedKnownDurationStops.append(stop)
            if stop[6] == -1:  # Case of unknown duration but joined
                self.joinedUnknownDurationStops.append(stop)
        self.knownDurationStopsNP = numpy.transpose(numpy.array(self.knownDurationStops))  # For time and columns accessible as self.stopsNP[0], self.stopsNP[1], ...
        self.unknownDurationStopsNP = numpy.transpose(numpy.array(self.unknownDurationStops))
        self.joinedKnownDurationStopsNP = numpy.transpose(numpy.array(self.joinedKnownDurationStops))  
        self.joinedUnknownDurationStopsNP = numpy.transpose(numpy.array(self.joinedUnknownDurationStops))  
        try:
            endTimesKnownDurationStops = self.knownDurationStopsNP[0] + self.knownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeKnownDurationStops = numpy.insert(self.knownDurationStopsNP[0], numpy.arange(1, len(self.knownDurationStopsNP[0]) + 1), endTimesKnownDurationStops)
            couplesXKnownDurationStops = numpy.insert(self.knownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.knownDurationStopsNP[1]) + 1), self.knownDurationStopsNP[1] - 0.5)
            self.graphKnownDurationStops = pyqtgraph.PlotDataItem(couplesTimeKnownDurationStops, couplesXKnownDurationStops, pen='g', connect='pairs')           
            self.startGraphKnownDurationStops = pyqtgraph.ScatterPlotItem(self.knownDurationStopsNP[0], self.knownDurationStopsNP[1], pen='g', brush='g')
            self.trajectoryGraph.addItem(self.graphKnownDurationStops)   # Duration of stops
            self.trajectoryGraph.addItem(self.startGraphKnownDurationStops)  # Start points of stops
        except:  # For empty list of known duration stops
            pass  
        try:
            endTimesUnknownDurationStops = self.unknownDurationStopsNP[0] + self.unknownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeUnknownDurationStops = numpy.insert(self.unknownDurationStopsNP[0], numpy.arange(1, len(self.unknownDurationStopsNP[0]) + 1), endTimesUnknownDurationStops)
            couplesXUnknownDurationStops = numpy.insert(self.unknownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.unknownDurationStopsNP[1]) + 1), self.unknownDurationStopsNP[1] - 0.5)
            self.graphUnknownDurationStops = pyqtgraph.PlotDataItem(couplesTimeUnknownDurationStops, couplesXUnknownDurationStops, pen='r', connect='pairs')
            self.startGraphUnknownDurationStops = pyqtgraph.ScatterPlotItem(self.unknownDurationStopsNP[0], self.unknownDurationStopsNP[1], pen='r', brush='r')
            self.trajectoryGraph.addItem(self.graphUnknownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startGraphUnknownDurationStops)  # Start points of stops
        except:  # For empty list of unknown duration stops
            pass
        try:
            endTimesJoinedKnownDurationStops = self.joinedKnownDurationStopsNP[0] + self.joinedKnownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeJoinedKnownDurationStops = numpy.insert(self.joinedKnownDurationStopsNP[0], numpy.arange(1, len(self.joinedKnownDurationStopsNP[0]) + 1), endTimesJoinedKnownDurationStops)
            couplesXJoinedKnownDurationStops = numpy.insert(self.joinedKnownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.joinedKnownDurationStopsNP[1]) + 1), self.joinedKnownDurationStopsNP[1] - 0.5)
            self.graphJoinedKnownDurationStops = pyqtgraph.PlotDataItem(couplesTimeJoinedKnownDurationStops, couplesXJoinedKnownDurationStops, pen='c', connect='pairs')    
            self.startGraphJoinedKnownDurationStops = pyqtgraph.ScatterPlotItem(self.joinedKnownDurationStopsNP[0], self.joinedKnownDurationStopsNP[1], pen='c', brush='c')
            self.trajectoryGraph.addItem(self.graphJoinedKnownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startGraphJoinedKnownDurationStops)  # Start points of stops
        except:  # For empty list of joined known duration stops
            pass
        try:
            endTimesJoinedUnknownDurationStops = self.joinedUnknownDurationStopsNP[0] + self.joinedUnknownDurationStopsNP[4]  # End time of stop = start time + duration
            couplesTimeJoinedUnknownDurationStops = numpy.insert(self.joinedUnknownDurationStopsNP[0], numpy.arange(1, len(self.joinedUnknownDurationStopsNP[0]) + 1), endTimesJoinedUnknownDurationStops)
            couplesXJoinedUnknownDurationStops = numpy.insert(self.joinedUnknownDurationStopsNP[1] - 0.5, numpy.arange(1, len(self.joinedUnknownDurationStopsNP[1]) + 1), self.joinedUnknownDurationStopsNP[1] - 0.5)
            self.graphJoinedUnknownDurationStops = pyqtgraph.PlotDataItem(couplesTimeJoinedUnknownDurationStops, couplesXJoinedUnknownDurationStops, pen='y', connect='pairs')    
            self.startGraphJoinedUnknownDurationStops = pyqtgraph.ScatterPlotItem(self.joinedUnknownDurationStopsNP[0], self.joinedUnknownDurationStopsNP[1], pen='y', brush='y')
            self.trajectoryGraph.addItem(self.graphJoinedUnknownDurationStops)  # Duration of stops
            self.trajectoryGraph.addItem(self.startGraphJoinedUnknownDurationStops)  # Start points of stops
        except:  # For empty list of joined unknown duration stops
            pass
        return
    
    def requestPrevious(self):  # GUI function to go back
        self.returnValue.emit(-1)
        self.trajectoryGraph.close()
        self.areaHistogram.close()
        self.speedHistogram.close()
        self.window.close()
        return 

    def ignoreResults(self):
        self.returnValue.emit(1)
        self.trajectoryGraph.close()
        self.areaHistogram.close()
        self.speedHistogram.close()
        self.window.close()
        return
    #### Video verification of trajectories and stops on film ####

    def createVerificationVideo(self):  # Create videos with trajectories and stops visible for tuning
        self.videoWindow = QtWidgets.QWidget()
        # Construct lists of stop points for display (time, x, area) for the three categories
        stopPointsList = []  # Known durations
        for stop in self.stops:  # FORMAT of items in stopList: absoluteTime, X, Y, area, duration, traj number, endType
            stopPointsList.append([stop[0], stop[1], stop[2], stop[3], stop[6]])  # First point
            for d in range(int(stop[4])):
                stopPointsList.append([stop[0] + d, stop[1], stop[2], stop[3], stop[6]])  # Subsequent points of the stop  
        stopPointsArray = numpy.array(stopPointsList) 
        # Open video
        options = QFileDialog.Options()  # Options for QFileDialog.QOpenFileName
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.avi);;All Files (*)", options=options)
        if filePath:
            videoAddress = filePath 
        capture = cv2.VideoCapture(videoAddress)        
        if not capture.isOpened():
            raise ValueError("Unable to open video file")        
        numFirstFrame = self.trajsNP[0, 0, 1]  # First image of first frame, in absolute time # The 1 avoids first NaN (to be verified)
        absoluteTimeColumns = self.trajsNP[:, 0, :]  # Form table of temporal columns
        numLastFrame = cupy.nanmax(absoluteTimeColumns)  # Max method of a table ignoring naN  # Last image of the last trajectory, in absolute time
        videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video properties #OK
        videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # OK
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        videoOutput = cv2.VideoWriter(filePath[:-4] + "Stops" + self.splitAddress[0][len(self.splitAddress[0]) - 1:] + ".avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (videoWidth, videoHeight))
        for n in range(int(numLastFrame - numFirstFrame)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, n + int(numFirstFrame)) 
            ret, frame = capture.read()
            if ret:
                conditionFrameN = self.trajsNP[:, 0, :] == float(n + int(numFirstFrame))
                conditionFrameN = cupy.reshape(conditionFrameN, (self.trajsNP.shape[0], 1, self.trajsNP.shape[2]))  # Desired format in argument
                conditionFrameN = cupy.tile(conditionFrameN, (1, 4, 1))  # Number of desired copies in argument               
                positionsBeadsFrameN = self.trajsNP[conditionFrameN]
                positionsBeadsFrameN = cupy.reshape(positionsBeadsFrameN, (int(positionsBeadsFrameN.shape[0] / 4), 4))
                listPositionsBeadsFrameN = positionsBeadsFrameN.tolist()  # OK        
                for pos in listPositionsBeadsFrameN:
                    cv2.circle(frame, (int(pos[1]), int(pos[2])), 2 * int(math.sqrt(pos[3] / 3.1416)), (0, 255, 255), -1)  # image, center position, radius, color, thickness (-1 = fill)                         
                conditionStopInFrame = areasStops[:, 0] == float(n + int(numFirstFrame))            
                positionsStopInFrame = conditionStopInFrame.nonzero()[0]
                if len(positionsStopInFrame) > 0:
                    for posArr in positionsStopInFrame.tolist():
                        if areasStops[posArr, 4] == float(1):  # Case of known duration
                            cv2.circle(frame, (int(areasStops[posArr, 1]), int(areasStops[posArr, 2])), 2 * int(math.sqrt(areasStops[posArr, 3] / 3.1416)), (0, 255, 0), -1)  # image, center position, radius, color, thickness (-1 = fill)
                        if areasStops[posArr, 4] == float(0):  # Case of unknown duration
                            cv2.circle(frame, (int(areasStops[posArr, 1]), int(areasStops[posArr, 2])), 2 * int(math.sqrt(areasStops[posArr, 3] / 3.1416)), (0, 0, 255), -1)  # image, center position, radius, color, thickness (-1 = fill)
                        if areasStops[posArr, 4] == float(-1):  # Case of unknown duration but joined
                            cv2.circle(frame, (int(areasStops[posArr, 1]), int(areasStops[posArr, 2])), 2 * int(math.sqrt(areasStops[posArr, 3] / 3.1416)), (255, 255, 0), -1)  # image, center position, radius, color, thickness (-1 = fill)
                        if areasStops[posArr, 4] == float(-2):  # Case of unknown duration joined
                            cv2.circle(frame, (int(areasStops[posArr, 1]), int(areasStops[posArr, 2])), 2 * int(math.sqrt(areasStops[posArr, 3] / 3.1416)), (255, 0, 0), -1)  # image, center position, radius, color, thickness (-1 = fill)
                videoOutput.write(frame)      
            else:
                break  # End of while
###################################################### Global GUI ##################################################
        
class GUI(QMainWindow):
    global guiBlocked
    global DURATION_MEASURE_SPEED, MIN_SPEED_FIT, THRESHOLD_STOP_X, THRESHOLD_STOP_Y, THRESHOLD_STOP_TEMP, DURATION_CHECK_STOP, DELAY_CHECK_SPEED, THRESHOLD_RETURN_BACK, MAX_MOVE_THRESHOLD_STOP_X, MAX_MOVE_THRESHOLD_STOP_Y, MAX_TEMP_JOIN_THRESHOLD, COEFF_AREA_PLUS, COEFF_AREA_MINUS, COEFF_MaxVMax, COEFF_MinVMin, MAX_AREA_HISTO, STEP_HISTO_AREAS, STEP_HISTO_SPEEDS, MIN_DISTANCE_TWO_BEADS_X, MIN_DISTANCE_TWO_BEADS_Y, TIME_CONVERSION_FACTOR, PIX_MICRON_CONVERSION_FACTOR, CLICK_PROXIMITY_THRESHOLD_X, CLICK_PROXIMITY_THRESHOLD_T
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(100, 100, 400, 460)
        self.setWindowTitle("PyStops")
        self._createMenuBar()  
        center = QWidget()  # Other widgets placed in a CentralWidget necessary for the menu bar
        self.setCentralWidget(center) 
        self.textZone = QPlainTextEdit(center)
        self.textZone.setGeometry(0, 0, 400, 300)             
    ##### Buttons to launch automated analysis ########
        buttonOpen = QPushButton(center)
        buttonOpen.setText("Automatic Processing")
        buttonOpen.clicked.connect(self.createAddressListAviAndLaunchTrajectoriesAndStops)  # Form: self.addressListAvi
        buttonOpen.setGeometry(0, 305, 400, 40)    
        buttonVerify = QPushButton(center)
        buttonVerify.setText("Verification")
        buttonVerify.clicked.connect(self.launchGUIVerification)
        buttonVerify.setGeometry(0, 350, 400, 40)      
        buttonReanalyze = QPushButton(center)
        buttonReanalyze.setText("Reanalyze")
        buttonReanalyze.clicked.connect(self.launchGUIReanalyze)
        buttonReanalyze.setGeometry(0, 395, 400, 40)               
        self.addressListAvi = [] 
        self.addressListTrajFiles = []  # List of addresses of trajectory sequences for stop analysis


    def _createMenuBar(self):    
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)     
    ##### Manual analysis menus (= pyTrajs and pyStops) ########
        manualAnalysisMenu = QMenu("&Separate Analyses", self)    
        self.actionLaunchTrajs = QAction(self)
        self.actionLaunchTrajs.triggered.connect(self.windowTrajsMan)  # Analyze the address list self.addressListAvi identical to that chosen by buttonOpen
        self.actionLaunchTrajs.setText("Trajectories")
        manualAnalysisMenu.addAction(self.actionLaunchTrajs)   
        self.actionLaunchStopsBatch = QAction(self)
        self.actionLaunchStopsBatch.triggered.connect(self.windowStopsBatch)  # Analyze the address list self.addressListTrajs chosen by buttonOpen
        self.actionLaunchStopsBatch.setText("Autonomous Stops")
        manualAnalysisMenu.addAction(self.actionLaunchStopsBatch)       
        self.actionLaunchStops = QAction(self)
        self.actionLaunchStops.triggered.connect(self.windowStopsMan)  # Analyze the address list self.addressListTrajs chosen by buttonOpen
        self.actionLaunchStops.setText("Stops Sequence by Sequence")
        manualAnalysisMenu.addAction(self.actionLaunchStops)       
        menuBar.addMenu(manualAnalysisMenu)  
    ##### Specific analysis menus: single video, analysis with backward compatibility... ########
        specificAnalysisMenu = QMenu("&Specific Analyses", self)
        menuBar.addMenu(specificAnalysisMenu） 
    ##### Default parameter settings menus ########
        parametersMenu = QMenu("&Parameters", self)    
        self.actionParamTrajs = QAction(self)
        self.actionParamTrajs.triggered.connect(self.windowParamTrajs)
        self.actionParamTrajs.setText("Trajectory Parameters")
        parametersMenu.addAction(self.actionParamTrajs)    
        self.actionParamStops = QAction(self)
        self.actionParamStops.triggered.connect(self.windowParamStops)
        self.actionParamStops.setText("Stop Parameters")
        parametersMenu.addAction(self.actionParamStops)          
        menuBar.addMenu(parametersMenu)
 
    ############ windows for parameter adjustment ############    
        #### 1) parameters for trajectory formation ####
        
    def windowParamTrajs(self): 
        self.parameterTrajectoryWindow = QtWidgets.QWidget()
        self.parameterTrajectoryWindow.setGeometry(120, 120, 400, 50) 
        self.parameterTrajectoryWindow.setWindowTitle("Trajectory Formation Parameters")   
        self.inputPERCENTILE = QLineEdit(self)
        self.inputMINMASS = QLineEdit(self)
        self.inputDIM_X = QLineEdit(self)
        self.inputDIM_Y = QLineEdit(self)       
        self.inputSEARCH_INTERVAL_Y = QLineEdit(self)  
        self.inputSEARCH_INTERVAL_X = QLineEdit(self)
        self.inputDEFAULT_SEARCH_RANGE_X = QLineEdit(self)
        self.inputDEFAULT_SEARCH_RANGE_Y = QLineEdit(self)
        self.inputMEMORY = QLineEdit(self)
        self.inputTHRESHOLD = QLineEdit(self)   
        self.inputPERCENTILE.setText(str(DEFAULT_PERCENTILE))
        self.inputMINMASS.setText(str(DEFAULT_MINMASS))
        self.inputDIM_X.setText(str(DEFAULT_DIMENSION_X))
        self.inputDIM_Y.setText(str(DEFAULT_DIMENSION_Y))
        self.inputSEARCH_INTERVAL_X.setText(str(DEFAULT_SEARCH_RANGE_X))
        self.inputSEARCH_INTERVAL_Y.setText(str(DEFAULT_SEARCH_RANGE_Y))
        self.inputDEFAULT_SEARCH_RANGE_X.setText(str(DEFAULT_SEPARATION_X))
        self.inputDEFAULT_SEARCH_RANGE_Y.setText(str(DEFAULT_SEPARATION_Y))
        self.inputMEMORY.setText(str(DEFAULT_MEMORY))
        self.inputTHRESHOLD.setText(str(DEFAULT_THRESHOLD))    
        layout = QVBoxLayout(self.parameterTrajectoryWindow)  
        layout.addWidget(QLabel("Luminance Percentile:"))
        layout.addWidget(self.inputPERCENTILE)
        layout.addWidget(QLabel("Minimum Luminance Mass:"))
        layout.addWidget(self.inputMINMASS)
        layout.addWidget(QLabel("Dimension in X:"))
        layout.addWidget(self.inputDIM_X)
        layout.addWidget(QLabel("Dimension in Y:"))
        layout.addWidget(self.inputDIM_Y)
        layout.addWidget(QLabel("Search Distance in X:"))
        layout.addWidget(self.inputSEARCH_INTERVAL_X)
        layout.addWidget(QLabel("Search Distance in Y:"))
        layout.addWidget(self.inputSEARCH_INTERVAL_Y)
        layout.addWidget(QLabel("Separation in X:"))
        layout.addWidget(self.inputDEFAULT_SEARCH_RANGE_X)
        layout.addWidget(QLabel("Separation in Y:"))
        layout.addWidget(self.inputDEFAULT_SEARCH_RANGE_Y)
        layout.addWidget(QLabel("Reconnect Memory:"))
        layout.addWidget(self.inputMEMORY)
        layout.addWidget(QLabel("Length Threshold for Saving:"))
        layout.addWidget(self.inputTHRESHOLD)
        def validateTrajectoryValues(): 
            global PERCENTILE, MINMASS, DIM_X, DIM_Y, SEARCH_RANGE_X, SEARCH_RANGE_Y, SEPARATION_X, SEPARATION_Y, MEMORY, THRESHOLD
            PERCENTILE = float(self.inputPERCENTILE.text())
            MINMASS = float(self.inputMINMASS.text())
            DIM_X = float(self.inputDIM_X.text())
            DIM_Y = float(self.inputDIM_Y.text())
            SEARCH_RANGE_X = float(self.inputSEARCH_INTERVAL_X.text())
            SEARCH_RANGE_Y = float(self.inputSEARCH_INTERVAL_Y.text())
            SEPARATION_X = float(self.inputDEFAULT_SEARCH_RANGE_X.text())
            SEPARATION_Y = float(self.inputDEFAULT_SEARCH_RANGE_Y.text())
            MEMORY = float(self.inputMEMORY.text())
            THRESHOLD = float(self.inputTHRESHOLD.text())
            self.parameterTrajectoryWindow.close()
        buttonValidate = QPushButton(self.parameterTrajectoryWindow)
        buttonValidate.setText("Validate Values")
        buttonValidate.clicked.connect(validateTrajectoryValues)
        layout.addWidget(buttonValidate)
        self.parameterTrajectoryWindow.show()

    #### 2) Stop Detection Parameters ############    

    def windowStopParameters(self): 
        global DURATION_MEASURE_SPEED, MIN_SPEED_FIT, THRESHOLD_STOP_X, THRESHOLD_STOP_Y, THRESHOLD_STOP_TEMP, DURATION_CHECK_STOP, DELAY_CHECK_SPEED, THRESHOLD_RETURN_BACK, MAX_MOVE_THRESHOLD_STOP_X, MAX_MOVE_THRESHOLD_STOP_Y, MAX_TEMP_JOIN_THRESHOLD, COEFF_AREA_PLUS, COEFF_AREA_MINUS, COEFF_MaxVMax, COEFF_MinVMin, MAX_AREA_HISTO, STEP_HISTO_AREAS, STEP_HISTO_SPEEDS, MIN_DISTANCE_TWO_BEADS_X, MIN_DISTANCE_TWO_BEADS_Y, TIME_CONVERSION_FACTOR, PIX_MICRON_CONVERSION_FACTOR, CLICK_PROXIMITY_THRESHOLD_X, CLICK_PROXIMITY_THRESHOLD_T    
        self.stopParamWindow = QtWidgets.QWidget()
        self.stopParamWindow.setGeometry(120, 120, 1000, 600) 
        self.setWindowTitle("Trajectory Formation Parameters")
        self.inputDURATION_MEASURE_SPEED = QLineEdit(self)
        self.inputMIN_SPEED_FIT = QLineEdit(self)
        self.inputTHRESHOLD_STOP_X = QLineEdit(self)
        self.inputTHRESHOLD_STOP_Y = QLineEdit(self)
        self.inputTHRESHOLD_STOP_TEMP = QLineEdit(self)
        self.inputDURATION_CHECK_STOP = QLineEdit(self)
        self.inputDELAY_CHECK_SPEED = QLineEdit(self)
        self.inputTHRESHOLD_RETURN_BACK = QLineEdit(self)
        self.inputMAX_MOVE_THRESHOLD_STOP_X = QLineEdit(self)
        self.inputMAX_MOVE_THRESHOLD_STOP_Y = QLineEdit(self)
        self.inputMAX_TEMP_JOIN_THRESHOLD = QLineEdit(self)
        self.inputCOEFF_AREA_PLUS = QLineEdit(self)
        self.inputCOEFF_AREA_MINUS = QLineEdit(self)
        self.inputCOEFF_MaxVMax = QLineEdit(self)
        self.inputCOEFF_MinVMin = QLineEdit(self)
        self.inputMAX_AREA_HISTO = QLineEdit(self)
        self.inputSTEP_HISTO_AREAS = QLineEdit(self)
        self.inputSTEP_HISTO_SPEEDS = QLineEdit(self)
        self.inputMIN_DISTANCE_TWO_BEADS_X = QLineEdit(self)
        self.inputMIN_DISTANCE_TWO_BEADS_Y = QLineEdit(self)
        self.inputTIME_CONVERSION_FACTOR = QLineEdit(self)
        self.inputPIX_MICRON_CONVERSION_FACTOR = QLineEdit(self)
        self.inputCLICK_PROXIMITY_THRESHOLD_X = QLineEdit(self)
        self.inputCLICK_PROXIMITY_THRESHOLD_T = QLineEdit(self)
        self.inputDURATION_MEASURE_SPEED.setText(str(DEFAULT_DURATION_MEASURE_SPEED))
        self.inputMIN_SPEED_FIT.setText(str(DEFAULT_MIN_SPEED_FIT))
        self.inputTHRESHOLD_STOP_X.setText(str(DEFAULT_THRESHOLD_STOP_X))
        self.inputTHRESHOLD_STOP_Y.setText(str(DEFAULT_THRESHOLD_STOP_Y))
        self.inputTHRESHOLD_STOP_TEMP.setText(str(DEFAULT_THRESHOLD_STOP_TEMP))
        self.inputDURATION_CHECK_STOP.setText(str(DEFAULT_DURATION_CHECK_STOP))
        self.inputDELAY_CHECK_SPEED.setText(str(DEFAULT_DELAY_CHECK_SPEED))
        self.inputTHRESHOLD_RETURN_BACK.setText(str(DEFAULT_THRESHOLD_RETURN_BACK))
        self.inputMAX_MOVE_THRESHOLD_STOP_X.setText(str(DEFAULT_MAX_MOVE_THRESHOLD_STOP_X))
        self.inputMAX_MOVE_THRESHOLD_STOP_Y.setText(str(DEFAULT_MAX_MOVE_THRESHOLD_STOP_Y))
        self.inputMAX_TEMP_JOIN_THRESHOLD.setText(str(DEFAULT_MAX_TEMP_JOIN_THRESHOLD))
        self.inputCOEFF_AREA_PLUS.setText(str(DEFAULT_COEFF_AREA_PLUS))
        self.inputCOEFF_AREA_MINUS.setText(str(DEFAULT_COEFF_AREA_MINUS))
        self.inputCOEFF_MaxVMax.setText(str(DEFAULT_COEFF_MaxVMax))
        self.inputCOEFF_MinVMin.setText(str(DEFAULT_COEFF_MinVMin))
        self.inputMAX_AREA_HISTO.setText(str(DEFAULT_MAX_AREA_HISTO))
        self.inputSTEP_HISTO_AREAS.setText(str(DEFAULT_STEP_HISTO_AREAS))
        self.inputSTEP_HISTO_SPEEDS.setText(str(DEFAULT_STEP_HISTO_SPEEDS))
        self.inputMIN_DISTANCE_TWO_BEADS_X.setText(str(DEFAULT_MIN_DISTANCE_TWO_BEADS_X))
        self.inputMIN_DISTANCE_TWO_BEADS_Y.setText(str(DEFAULT_MIN_DISTANCE_TWO_BEADS_Y))
        self.inputTIME_CONVERSION_FACTOR.setText(str(DEFAULT_TIME_CONVERSION_FACTOR))
        self.inputPIX_MICRON_CONVERSION_FACTOR.setText(str(DEFAULT_PIX_MICRON_CONVERSION_FACTOR))
        self.inputCLICK_PROXIMITY_THRESHOLD_X.setText(str(DEFAULT_CLICK_PROXIMITY_THRESHOLD_X))
        self.inputCLICK_PROXIMITY_THRESHOLD_T.setText(str(DEFAULT_CLICK_PROXIMITY_THRESHOLD_T))
        self.inputPIXEL_MICRON_CONV_FACTOR.setText(str(DEFAULT_PIXEL_MICRON_CONV_FACTOR))
        self.inputCLICK_PROXIMITY_THRESHOLD_X.setText(str(DEFAULT_CLICK_PROXIMITY_THRESHOLD_X))
        self.CLICK_PROXIMITY_THRESHOLD_T.setText(str(DEFAULT_CLICK_PROXIMITY_THRESHOLD_T))
        layoutLinesAndButton = QVBoxLayout()
        layoutColumnsH = QHBoxLayout() 
        layoutLeft = QVBoxLayout() 
        layoutRight = QVBoxLayout()   
        layoutLeft.addWidget(QLabel("Duration for speed measurement (frame)"))
        layoutLeft.addWidget(self.inputDURATION_MEASURE_SPEED)
        layoutLeft.addWidget(QLabel("Minimum speed for histogram fit (pixel/frame)"))
        layoutLeft.addWidget(self.inputMIN_SPEED_FIT)
        layoutLeft.addWidget(QLabel("Threshold distance in X for defining stops (pixel)"))
        layoutLeft.addWidget(self.inputSTOP_THRESHOLD_X)
        layoutLeft.addWidget(QLabel("Threshold distance in Y for defining stops (pixel)"))
        layoutLeft.addWidget(self.inputSTOP_THRESHOLD_Y)
        layoutLeft.addWidget(QLabel("Delay for defining stops (frame)"))
        layoutLeft.addWidget(self.inputSTOP_THRESHOLD_TEMP)
        layoutLeft.addWidget(QLabel("Duration for stop verification (frame)"))
        layoutLeft.addWidget(self.inputDURATION_CHECK_STOP)                    
        layoutLeft.addWidget(QLabel("Verification delay around stops (frame)"))
        layoutLeft.addWidget(self.inputDELAY_CHECK_SPEED)                    
        layoutLeft.addWidget(QLabel("Max backward displacement of centroid at stop start (pixel)"))
        layoutLeft.addWidget(self.inputBACKWARD_RETURN_THRESHOLD)                    
        layoutLeft.addWidget(QLabel("Max displacement in X during a stop (pixel)"))
        layoutLeft.addWidget(self.inputMAX_MOVEMENT_THRESHOLD_STOP_X)                  
        layoutLeft.addWidget(QLabel("Max displacement in Y during a stop (pixel)"))
        layoutLeft.addWidget(self.inputMAX_MOVEMENT_THRESHOLD_STOP_Y)
        layoutLeft.addWidget(QLabel("Max delay between two stop segments for reconnecting (frame)"))
        layoutLeft.addWidget(self.inputMAX_TEMP_SEP_RECONNECT_STOP)                    
        layoutLeft.addWidget(QLabel("Area tolerance coefficient (+)"))
        layoutLeft.addWidget(self.inputCOEFF_AREA_PLUS)
        layoutRight.addWidget(QLabel("Area tolerance coefficient (-)"))
        layoutRight.addWidget(self.inputCOEFF_AREA_MINUS)                
        layoutRight.addWidget(QLabel("Speed tolerance coefficient (+)"))
        layoutRight.addWidget(self.inputCOEFF_MaxVIncrease)                    
        layoutRight.addWidget(QLabel("Speed tolerance coefficient (-)"))
        layoutRight.addWidget(self.inputCOEFF_MinVDecrease)                    
        layoutRight.addWidget(QLabel("Max value of the area histogram (pixel²)"))
        layoutRight.addWidget(self.inputMAX_AREA_HISTO) 
        layoutRight.addWidget(QLabel("Step size of area histogram (pixel²)"))
        layoutRight.addWidget(self.inputSTEP_HISTO_AREAS)    
        layoutRight.addWidget(QLabel("Step size of speed histogram (pixel/step)"))
        layoutRight.addWidget(self.inputSTEP_HISTO_SPEEDS) 
        layoutRight.addWidget(QLabel("Min distance in X between 2 bead centroids to ensure no contact during stop (pixel)"))
        layoutRight.addWidget(self.inputMIN_DISTANCE_TWO_BALLS_X)
        layoutRight.addWidget(QLabel("Min distance in Y between 2 bead centroids to ensure no contact during stop (pixel)"))
        layoutRight.addWidget(self.inputMIN_DISTANCE_TWO_BALLS_Y) 
        layoutRight.addWidget(QLabel("Time conversion factor (seconds/frame)"))
        layoutRight.addWidget(self.inputTEMP_CONVERSION_FACTOR)
        layoutRight.addWidget(QLabel("Spatial conversion factor (µm/pixel)"))
        layoutRight.addWidget(self.inputPIXEL_MICRON_CONV_FACTOR) 
        layoutRight.addWidget(QLabel("Proximity threshold in X for click in trajectory GUI (pixel)"))
        layoutRight.addWidget(self.inputCLICK_PROXIMITY_THRESHOLD_X)
        layoutRight.addWidget(QLabel("Proximity threshold in time for click in trajectory GUI (pixel)"))
        layoutRight.addWidget(self.CLICK_PROXIMITY_THRESHOLD_T)    
        def validateStopValues(): 
            global DURATION_MEASURE_SPEED, MIN_SPEED_FIT, THRESHOLD_STOP_X, THRESHOLD_STOP_Y, THRESHOLD_STOP_TEMP, DURATION_CHECK_STOP, DELAY_CHECK_SPEED, THRESHOLD_RETURN_BACK, MAX_MOVE_THRESHOLD_STOP_X, MAX_MOVE_THRESHOLD_STOP_Y, MAX_TEMP_JOIN_THRESHOLD, COEFF_AREA_PLUS, COEFF_AREA_MINUS, COEFF_MaxVMax, COEFF_MinVMin, MAX_AREA_HISTO, STEP_HISTO_AREAS, STEP_HISTO_SPEEDS, MIN_DISTANCE_TWO_BEADS_X, MIN_DISTANCE_TWO_BEADS_Y, TIME_CONVERSION_FACTOR, PIX_MICRON_CONVERSION_FACTOR, CLICK_PROXIMITY_THRESHOLD_X, CLICK_PROXIMITY_THRESHOLD_T    
            DURATION_MEASURE_SPEED = float(self.inputDURATION_MEASURE_SPEED.text())
            MIN_SPEED_FIT = float(self.inputMIN_SPEED_FIT.text()) 
            THRESHOLD_STOP_X = float(self.inputTHRESHOLD_STOP_X.text()) 
            THRESHOLD_STOP_Y = float(self.inputTHRESHOLD_STOP_Y.text()) 
            THRESHOLD_STOP_TEMP = float(self.inputTHRESHOLD_STOP_TEMP.text()) 
            DURATION_CHECK_STOP = float(self.inputDURATION_CHECK_STOP.text()) 
            DELAY_CHECK_SPEED = float(self.inputDELAY_CHECK_SPEED.text()) 
            THRESHOLD_RETURN_BACK = float(self.inputTHRESHOLD_RETURN_BACK.text()) 
            MAX_MOVE_THRESHOLD_STOP_X = float(self.inputMAX_MOVE_THRESHOLD_STOP_X.text()) 
            MAX_MOVE_THRESHOLD_STOP_Y = float(self.inputMAX_MOVE_THRESHOLD_STOP_Y.text()) 
            MAX_TEMP_JOIN_THRESHOLD = float(self.inputMAX_TEMP_JOIN_THRESHOLD.text()) 
            COEFF_AREA_PLUS = float(self.inputCOEFF_AREA_PLUS.text()) 
            COEFF_AREA_MINUS = float(self.inputCOEFF_AREA_MINUS.text()) 
            COEFF_MaxVMax = float(self.inputCOEFF_MaxVMax.text()) 
            COEFF_MinVMin = float(self.inputCOEFF_MinVMin.text()) 
            MAX_AREA_HISTO = float(self.inputMAX_AREA_HISTO.text()) 
            STEP_HISTO_AREAS = float(self.inputSTEP_HISTO_AREAS.text()) 
            STEP_HISTO_SPEEDS = float(self.inputSTEP_HISTO_SPEEDS.text()) 
            MIN_DISTANCE_TWO_BEADS_X = float(self.inputMIN_DISTANCE_TWO_BEADS_X.text()) 
            MIN_DISTANCE_TWO_BEADS_Y = float(self.inputMIN_DISTANCE_TWO_BEADS_Y.text()) 
            TIME_CONVERSION_FACTOR = float(self.inputTIME_CONVERSION_FACTOR.text()) 
            PIX_MICRON_CONVERSION_FACTOR = float(self.inputPIX_MICRON_CONVERSION_FACTOR.text()) 
            CLICK_PROXIMITY_THRESHOLD_X = float(self.inputCLICK_PROXIMITY_THRESHOLD_X.text()) 
            CLICK_PROXIMITY_THRESHOLD_T = float(self.inputCLICK_PROXIMITY_THRESHOLD_T.text())    
            self.stopParamWindow.close()
            print(DURATION_MEASURE_SPEED)
        buttonValidate = QPushButton(self.stopParamWindow)
        buttonValidate.setText("Validate Values")
        buttonValidate.clicked.connect(validateStopValues)
        layoutColumnsH.addLayout(layoutLeft)
        layoutColumnsH.addLayout(layoutRight)
        layoutLinesAndButton.addLayout(layoutColumnsH)
        layoutLinesAndButton.addWidget(buttonValidate)
        self.stopParamWindow.setLayout(layoutLinesAndButton)
        self.stopParamWindow.show()
        
        
    ############ Global Autonomous Control Functions ##################

    def createAddressListAviAndLaunchTrajectoriesAndStops(self):  # Form a list of films from addresses in a starting folder (explore the structure)
        self.textZone.insertPlainText("Opening a folder \n")
        selectedFolder = QFileDialog.getExistingDirectory(self, "Choose a Folder of Films", options=QFileDialog.ShowDirsOnly)        
        foldersToExplore = [selectedFolder]        
        def openFolder(folder):
            for root, directories, files in os.walk(folder):
                for file in files: 
                    if file.endswith(".avi"):
                        self.addressListAvi.append(os.path.join(root, file))
                for d in directories:  # Add discovered folders to the list of folders to explore
                    if os.path.join(root, d) not in foldersToExplore:
                        foldersToExplore.append(os.path.join(root, d))                                       
        for folder in foldersToExplore: 
            openFolder(folder)  # Iterate over all subfolders of the initial folder in search of .avi files          
        for address in self.addressListAvi:
            self.textZone.insertPlainText(str(address) + '\n')           
        self.launchAutonomousAnalysis()     
           
    def launchAutonomousAnalysis(self):         
        for filmAddress in self.addressListAvi:
            trajectoryGroupsForOneVideo = self.createGroupsForTrajectoriesInOneVideo(filmAddress)  # Cut into sequences and then form trajectories 
            print(len(trajectoryGroupsForOneVideo))
            self.analyzeTrajectoryForOneFilmAuto(trajectoryGroupsForOneVideo, filmAddress)  # Detect stops for each sequence
    def analyzeTrajectoryForOneFilmAuto(self, resultsForOneFilm, filmAddress):  # resultsListOpening is a list of lists of numpy tables adapted for initial saving in .txt: 1 table per traj; 1 list of tables per sequence  # address is the film address: the .avi extension needs to be removed
        # Boolean to block direct closure of the display
        global guiBlocked 
        guiBlocked = True   
        for n, resultForSequence in enumerate(resultsForOneFilm):  # resultSequencesForFilm is a list of numpy tables: 1 table per traj 
            sequenceResultTuple = createInternalTrajTable(resultForSequence)  # Conversion using createInternalTrajTable for stop detection formatting
            trajsNP = sequenceResultTuple[0]
            startFrame = sequenceResultTuple[1]        
            areaResults = sortByAreas(trajsNP)  # areaResultTuple: TrajsManipTrAires, minArea, maxArea, binsHistoA, barsHistoA, fitResults
            speedResults = detectSpeedThresholdsAndMeasureDistance(areaResults[0])  # speedResultTuple: minSpeed, maxSpeed, distanceAfterSortingV, binsHistoV, barsHistoV, fitResults
            stopPositions = detectSortedStops(areaResults[0], areaResults[1], areaResults[2], speedResults[0], speedResults[1])  # Arguments: TrajsManipTrAires, minArea, maxArea, minSpeed, maxSpeed
            allStopsTuple = createStopList(areaResults[0], startFrame, stopPositions[0], stopPositions[1], stopPositions[2], stopPositions[3])  # Arguments: TrajsManipArrets, start frame number, startStopPositions, endStopPositions, positionsWithoutConditions
            reshapedTrajsForPlot = reshapePlotUniqueTrajectory(areaResults[0])  
            address = filmAddress[:-4] + os.sep + str(n + 1) + os.sep + "Trajectories.txt"  # The address here is the initial address of the film: adaptation of the address for plotGraph which requires the trajectory file address in .txt 
            print(address)
            self.currentPlot = plotTrajs(address, trajsNP, startFrame, reshapedTrajsForPlot, allStopsTuple, areaResults[1], areaResults[2], areaResults[3], areaResults[4], areaResults[5], speedResults[0], speedResults[1], speedResults[2], speedResults[3], speedResults[4], speedResults[5])
            self.currentPlot.plotGraph()  # Display function retained for saving as an image, boolean to indicate absence of direct display and direct closure


    ############ Functions for Trajectory Formation Belonging to the GUI ############

        #### 1) Function to process a complete video for subsequent analysis of autonomous or manual stops ############    

    def createGroupsForTrajectoriesInOneVideo(self, address):  # Basic analysis function for an .avi film: cut the film and then form trajectories
        filmName = os.path.basename(address)[:-4]  # File name without the .avi extension
        outputFolder = os.path.join(os.path.dirname(address), filmName)  # Create the output folder name
        os.mkdir(outputFolder)        
        print("Fragmenting")
        self.textZone.insertPlainText("Fragmenting into cycles\n")
        acquisitions = splitVideo(address, AVERAGE_THRESHOLD)  # Cut the film between the black sequences; AVERAGE_THRESHOLD value is the threshold of gray level to consider the image as a separator or acquisition image
        acquisitionList = formBoundPairs(acquisitions)  # Extract the numbers of the separator images from the 7 acquisitions
        self.textZone.insertPlainText("Starting cycle processing\n")
        trajectoryCounts = ""
        listsOfSubTablesForOneFilm = []   
        for i, acqu in enumerate(acquisitionList):  # Process acquisitions and save in text form
            print(acqu)
            resAcqu = processAcquisition(address, acqu)  # Return: list of sub-tables, resultTable (for pyStops), verification text, number of trajs in acqu
            if len(resAcqu[0]) > 0:  # Case of non-empty sequence (empty if sequences are very brief below the minimum length threshold for recording)
                listsOfSubTablesForOneFilm.append(resAcqu[0])  # List of lists of subTables
                self.textZone.insertPlainText("1 cycle processed\n")            
                j = i + 1 
                filePath = os.path.join(outputFolder, f'{j}/Trajectories.txt')
                trajectoryCounts += "\t" + str(j) + " : " + str(resAcqu[1]) + "\n"  # Form a list of trajs by acquisition    
                saveResultsAsText(resAcqu[0], filePath)  # Save results as text files    
        self.textZone.insertPlainText("Cycles processed\n")
        # Create a file to save parameters
        parameters = [
            "Dimension x: " + str(DIM_X),
            "Dimension y: " + str(DIM_Y),
            "Search Distance x: " + str(SEARCH_RANGE_X),
            "Search Distance y: " + str(SEARCH_RANGE_Y),
            "Memory: " + str(MEMORY),
            "Threshold: " + str(THRESHOLD)
        ]   
        self.textZone.insertPlainText("Trajectories:\n" + trajectoryCounts)    
        with open(os.path.join(outputFolder, 'Parameters.txt'), 'w') as f:
            f.write("Trajectories:\n" + trajectoryCounts)  # Save number of trajs 
            for line in parameters:  # Save used parameters
                f.write(f"{line}\n")                  
        self.textZone.insertPlainText("Trajectory formation completed\n")       
        return listsOfSubTablesForOneFilm  # One set of sub-tables per sequence; the list of these sets is the result of a non-fragmented video
        
    #### 2) Manual Functions ############

    def createAnalysisAddressListTrajs(self):  # Batch opening of .avi films for trajectories only (replaces pyTrajs)
        self.textZone.insertPlainText("Opening a folder of films for trajectories \n")
        self.addressListAviTrajs = []
        selectedFolder = QFileDialog.getExistingDirectory(self, "Choose a Folder of Films", options=QFileDialog.ShowDirsOnly)        
        foldersToExplore = [selectedFolder]            
        def openFolder(folder):
            for root, directories, files in os.walk(folder):  # Iterate over the structure
                for file in files: 
                    if file.endswith(".avi"):
                        self.addressListAviTrajs.append(os.path.join(root, file))
                for d in directories:  # Add discovered folders to the list of folders to explore
                    if os.path.join(root, d) not in foldersToExplore:
                        foldersToExplore.append(os.path.join(root, d))                                        
        for folder in foldersToExplore: 
            openFolder(folder)  # Iterate over all subfolders of the initial folder in search of .avi files      
        for address in self.addressListAviTrajs:
            self.textZone.insertPlainText(str(address) + '\n')      
        for address in self.addressListAviTrajs:
            print("Trajectories only")
            print(address)
            self.createGroupsForTrajectoriesInOneVideo(address)
    def windowManualTrajs(self):
        self.manualTrajsWindow = QtWidgets.QWidget()
        self.manualTrajsWindow.setGeometry(420, 120, 300, 50)         
        self.manualTrajsWindow.setWindowTitle("Trajectory Formation")      
        buttonOpen = QPushButton(self.manualTrajsWindow)
        buttonOpen.setText("Folder of Films to Process")
        buttonOpen.clicked.connect(self.createAnalysisAddressListTrajs)  # Form: self.addressListAvi
        buttonOpen.setGeometry(0, 5, 300, 40)     
        self.manualTrajsWindow.show()
        self.manualTrajsWindow.setFocus()
    
    def openFile(self):  # Opening and directly forming trajectories from a single film from GUI (replaces pyTrajs)
        self.textZone.insertPlainText("Directly opening a film \n")
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, "Open a Film", "", "Video Files (*.avi);;All Files (*)", options=options)   
        if filePath:
            print(filePath)
            self.address = filePath   
            self.batchAnalysis = False 
        self.analyzeVideo()
        
        
    ############ Stop Detection Functions from the GUI ############   

    def createAndAnalyzeAddressListStops(self):  # Create the list of trajectory files for each sequence for isolated stop analysis (i.e., without trajectory formation), with or without classic verification GUI
        global guiBlocked  # Boolean to block direct closure of the display                                      #(i.e., for all sequences in a list), based on manual selection of a folder containing trajectory sequences
        self.addressListTrajFiles = []
        self.textZone.insertPlainText("Opening a folder of trajectories for stop detection \n")
        selectedFolder = QFileDialog.getExistingDirectory(self, "Choose a Folder", options=QFileDialog.ShowDirsOnly)        
        foldersToExplore = [selectedFolder]           
        def openFolder(folder):
            for root, directories, files in os.walk(folder):
                for file in files: 
                    if file == "Trajectories.txt":
                        self.addressListTrajFiles.append(os.path.join(root, file))
                for d in directories:  # Add discovered folders to the list of folders to explore
                    if os.path.join(root, d) not in foldersToExplore:  # Avoid repeating folder openings
                        foldersToExplore.append(os.path.join(root, d))                    
        for folder in foldersToExplore: 
            openFolder(folder)  # Iterate over all subfolders of the initial folder in search of trajectory files  
        
        for address in self.addressListTrajFiles:
            self.textZone.insertPlainText(str(address) + '\n')    
        self.analysisCount = 0  # Used for backtracking
        print("Address after creating the list")
        print(self.addressListTrajFiles[self.analysisCount])
        self.analyzeStopsForOneSequence(self.addressListTrajFiles[self.analysisCount])

    def analyzeStopsForOneSequence(self, address):  # Launch stop analysis for a sequence followed by immediate manual verification GUI
        global guiBlocked  # Boolean to block direct closure of the display
        # Called by the direct manual analysis GUI createAndAnalyzeAddressListStops above AND by the verification GUI below if verification is chosen for certain sequences            
        resOpening = openTrajectoryFile(address)
        trajsNP = resOpening[0]
        startFrame = resOpening[1]        
        areaResults = sortByAreas(trajsNP)  # areaResultTuple: TrajsManipTrAires, minArea, maxArea, binsHistoA, barsHistoA, fitResults
        speedResults = detectSpeedThresholdsAndMeasureDistance(areaResults[0])  # speedResultTuple: vMin, vMax, distanceAfterSortingV, binsHistoV, barsHistoV, fitResults
        stopPositions = detectSortedStops(areaResults[0], areaResults[1], areaResults[2], speedResults[0], speedResults[1])  # Arguments: TrajsManipTrAires, minArea, maxArea, minSpeed, maxSpeed
        allStopsTuple = createStopList(areaResults[0], startFrame, stopPositions[0], stopPositions[1], stopPositions[2], stopPositions[3])  # Arguments: TrajsManipArrets, startStopPositions, endStopPositions, positionsWithoutConditions
        reshapedTrajsForPlot = reshapePlotUniqueTrajectory(areaResults[0])    
        self.currentPlot = plotTrajs(address, trajsNP, startFrame, reshapedTrajsForPlot, allStopsTuple, areaResults[1], areaResults[2], areaResults[3], areaResults[4], areaResults[5], speedResults[0], speedResults[1], speedResults[2], speedResults[3], speedResults[4], speedResults[5])
        self.currentPlot.returnValue.connect(self.incrementReturn)
        self.currentPlot.plotGraph()
    
    def incrementReturn(self, returnIncrement):
        global guiBlocked
        self.analysisCount += returnIncrement
        if self.analysisCount < len(self.addressListTrajFiles):            
            self.analyzeStopsForOneSequence(self.addressListTrajFiles[self.analysisCount])
        else:
            self.analysisCount = 0
            return
            
    def windowStopsBatch(self):  # Launch stop analysis on a folder of trajectories WITHOUT verification by GUI
        global guiBlocked
        guiBlocked = True
        self.batchStopsWindow = QtWidgets.QWidget()
        self.batchStopsWindow.setGeometry(420, 120, 300, 50)         
        self.batchStopsWindow.setWindowTitle("Stop Detection in a Set of Trajectory Folders.")       
        buttonOpen = QPushButton(self.batchStopsWindow)
        buttonOpen.setText("Folder of Trajectory Folders to Process")
        buttonOpen.clicked.connect(self.createAndAnalyzeAddressListStops) 
        buttonOpen.setGeometry(0, 5, 300, 40)
        self.batchStopsWindow.show()
        self.batchStopsWindow.setFocus()
        
    def windowStopsMan(self):  # Launch stop analysis on a folder of trajectories WITH verification by GUI
        global guiBlocked
        guiBlocked = False
        self.manualStopsWindow = QtWidgets.QWidget()
        self.manualStopsWindow.setGeometry(420, 120, 300, 50)         
        self.manualStopsWindow.setWindowTitle("Stop Detection with Control")       
        buttonOpen = QPushButton(self.manualStopsWindow)
        buttonOpen.setText("Folder of Trajectories to Process")
        buttonOpen.clicked.connect(self.createAndAnalyzeAddressListStops) 
        buttonOpen.setGeometry(0, 5, 300, 40)   
        self.manualStopsWindow.show()
        self.manualStopsWindow.setFocus()

        ####  3) #### Analysis of Trajectories for Backward Compatibility ########        
        
        # See pyStops !!!!!!!!!!!!!!!

    ############ Functions for quick verification of IMAGES of histograms, trajectories, and stops in the GUI after using autonomous functions
    # Then re-verification by GUI of ACTIVE histograms and ACTIVE graphs of trajectories and stops ############
     
    def launchGUIVerification(self): 
        self.addressListTrajsToVerify = []  # List of addresses of trajectories with stop results to be visually verified through the three graphs
        self.verificationWindow = QtWidgets.QWidget()  # Window with button to choose folder to verify
        self.verificationWindow.setGeometry(420, 120, 300, 140)         
        self.verificationWindow.setWindowTitle("Verification of Analyses")   
        buttonOpen = QPushButton(self.verificationWindow)
        buttonOpen.setText("Folder of Analyses to Verify")
        buttonOpen.clicked.connect(self.createVerifyPNGList)  # Form: self.addressTupleListForVerificationImages
        buttonOpen.setGeometry(0, 5, 300, 40)
        buttonSaveReanalyzeLater = QPushButton(self.verificationWindow)
        buttonSaveReanalyzeLater.setText("Save Choice of Sequences to Reanalyze")
        buttonSaveReanalyzeLater.clicked.connect(self.saveListOfSequencesToVerify)  # Form: self.addressTupleListForVerificationImages
        buttonSaveReanalyzeLater.setGeometry(0, 50, 300, 40)
        buttonReanalyze = QPushButton(self.verificationWindow)
        buttonReanalyze.setText("Immediate Reanalysis of Selected Sequences")
        buttonReanalyze.clicked.connect(self.launchRecalculateStopsWithManualVerifications)  # Form: self.addressTupleListForVerificationImages
        buttonReanalyze.setGeometry(0, 100, 300, 40)
        self.verificationWindow.show()
        self.verificationWindow.setFocus()
        
    def createVerifyPNGList(self):
        fVP = self.visualVerifyPNGList(self)  
        fVP.launchVisual()
   
    class VisualVerifyPNGList(QWidget):  # Class for visualization, creates a window with the three graphs for areas, speeds, and trajectories with control buttons
        def __init__(self, gui):
            super().__init__()
            folderToVerify = QFileDialog.getExistingDirectory(None, "Choose a Folder")  
            foldersToExplore = [folderToVerify]             
            gui.addressListTrajFiles.clear()  # Recreate the (empty) list of addresses for trajectory analysis of stops
            self.addressListTrajsToVerify = []  # List of addresses of trajectories with stop results to be verified by visualizing the three graphs
            self.tupleAddressesImagesVerifyList = []  # List of tuples where each contains the three addresses of the graphs for areas, speeds, and trajectories to display for verification
            self.addressListTrajsForReanalysis = []  # List of addresses to reanalyze using integrated pyStops from VisualVerifyPNGList
            def openFolder(folder):  # Create list of verification images (areas, speeds, and trajectories)   
                for root, directories, files in os.walk(folder):
                    for file in files: 
                        if file[len(file) - 9:] == "Aires.png":
                            self.addressListTrajsToVerify.append(os.path.join(root, file[: -9]) + os.sep + "Trajectoires.txt")
                            self.tupleAddressesImagesVerifyList.append((os.path.join(root, file), 
                                                                         os.path.join(root, file[: -9]) + "Vitesses.png", 
                                                                         os.path.join(root, file[: -9] + "Trajectoires.png")))
                    for d in directories:  # Add discovered folders to the list of folders to explore
                        if os.path.join(root, d) not in foldersToExplore:  # Avoid repeating the opening of the folder
                            foldersToExplore.append(os.path.join(root, d))                         
            for folder in foldersToExplore: 
                openFolder(folder)  # Iterate over all subfolders of the initial folder in search of .png files        
            # Window with the three graphs for areas, speeds, and trajectories with control buttons
            print((self.addressListTrajsToVerify))
            print((self.tupleAddressesImagesVerifyList))     
            self.setGeometry(0, 0, 1400, 1000)         
            self.setWindowTitle("Analysis Visualization")          
            self.labelAreas = QLabel(self)  # QLabel for displaying images or text             
            self.labelSpeeds = QLabel(self)  # QLabel for displaying images or text
            self.labelTrajs = QLabel(self)  # QLabel for displaying images or text        
            self.n = 0             
            self.loadImages()
            def validate():
                if self.n >= len(self.addressListTrajsToVerify):  # Condition: number of clicks less than or equal to the size of the list of addresses to verify
                    self.close()
                    if len(self.addressListTrajsForReanalysis) > 0: 
                        gui.addressListTrajFiles = self.addressListTrajsForReanalysis  # addressListTrajFiles is the list of sequences to open            
                self.n += 1
                if self.n < len(self.addressListTrajsToVerify):  # Continue displaying if the list to verify has not been fully reviewed 
                    self.loadImages()
                else: 
                    self.close()
                    gui.addressListTrajFiles = self.addressListTrajsForReanalysis
            self.buttonValidate = QPushButton(self)
            self.buttonValidate.setText("Validate") 
            self.buttonValidate.setStyleSheet("background-color: lime")
            self.buttonValidate.setFont(QFont('Arial', 17))
            self.buttonValidate.setGeometry(0, 0, 1400, 50)
            self.buttonValidate.clicked.connect(validate)
            def keepForVerification():
                if self.n >= len(self.tupleAddressesImagesVerifyList):  # Condition: number of clicks less than or equal to the size of the list of addresses to verify
                    self.close()
                    gui.addressListTrajFiles = self.addressListTrajsForReanalysis
                else:
                    self.addressListTrajsForReanalysis.append(self.addressListTrajsToVerify[self.n])           
                self.n += 1
                if self.n < len(self.tupleAddressesImagesVerifyList):  # Continue displaying if the list to verify has not been fully reviewed
                    self.loadImages()
                else: 
                    self.close()
                    gui.addressListTrajFiles = self.addressListTrajsForReanalysis            
            self.buttonReplay = QPushButton(self)
            self.buttonReplay.setText("For Verification")
            self.buttonReplay.setStyleSheet("background-color: orangered")
            self.buttonReplay.setFont(QFont('Arial', 17))
            self.buttonReplay.setGeometry(0, 50, 1400, 50)
            self.buttonReplay.clicked.connect(keepForVerification)
        def loadImages(self):  # A separate function for creating QLabels and identifying/loading the images to be loaded seems necessary for refreshing the display
            self.addressTuple = self.tupleAddressesImagesVerifyList[self.n]            
            self.pixmapAreas = QPixmap(self.addressTuple[0])  # Get the image
            self.labelAreas.setPixmap(self.pixmapAreas)  
            self.labelAreas.setScaledContents(True) 
            self.labelAreas.setGeometry(0, 100, 700, 400)     
            self.pixmapSpeeds = QPixmap(self.addressTuple[1])  # Get the image
            self.labelSpeeds.setPixmap(self.pixmapSpeeds)   
            self.labelSpeeds.setScaledContents(True)
            self.labelSpeeds.setGeometry(700, 100, 700, 400)   
            self.pixmapTrajs = QPixmap(self.addressTuple[2])  # Get the image
            self.labelTrajs.setPixmap(self.pixmapTrajs)    
            self.labelTrajs.setScaledContents(True)
            self.labelTrajs.setGeometry(0, 500, 1400, 500) 
        def launchVisual(self): 
            self.show()    
            
    def saveListOfSequencesToVerify(self):
        saveFolderForVerificationFiles = QFileDialog.getExistingDirectory(None, "Choose a Folder to Save Verifications")  
        saveAddressForVerificationFile = saveFolderForVerificationFiles + os.sep + "SequencesToVerify.txt"
        fileForVerificationSave = open(saveAddressForVerificationFile, "w")
        outputText = ""
        for addressToSave in self.addressListTrajFiles:
            outputText += addressToSave + "\n"            
        fileForVerificationSave.write(outputText)

    def launchRecalculateStopsWithManualVerifications(self):  # Launch from GUI verification on images of graphs and histograms; uses analyzeStopsForOneSequence and its return
        global guiBlocked
        guiBlocked = False  # Re-enable GUI
        self.analysisCount = 0         
        self.analyzeStopsForOneSequence(self.addressListTrajFiles[self.analysisCount])
        
    def launchGUIReanalyze(self):  # Launch actual reanalysis from the saved list
        global guiBlocked
        guiBlocked = False  # Re-enable GUI
        fileNameAddressesToReanalyze = QFileDialog.getOpenFileName(self, 'File of Addresses to Reanalyze')
        fileAddressesToReanalyze = open(fileNameAddressesToReanalyze[0], 'r')  # QFileDialog.getOpenFileName returns a tuple
        addressesToReanalyze = fileAddressesToReanalyze.readlines()
        processedAddressesToReanalyze = []        
        for address in addressesToReanalyze:
            address = address[:-1]  # Remove the \n; NOTE: counts as a single character
            words = re.split('[/\\\\]', address)  # Split by / and \\
            processedAddress = words[0]  # Avoid adding the first separator
            for word in words[1:]:  # Avoid adding the first separator
                processedAddress = processedAddress + os.sep + word
            processedAddressesToReanalyze.append(processedAddress)    
        self.analysisCount = 0
        self.analyzeStopsForOneSequence(processedAddressesToReanalyze[self.analysisCount])      
if __name__ == "__main__":  # Necessary because batch function of Trackpy relaunches GUIs for each core (unknown reason)
    app = QApplication(sys.argv)
    UI = GUI()
    UI.show()
    sys.exit(app.exec_())







