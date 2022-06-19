from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np 
import shelve
from os.path import exists

max_value = 255
max_value_H = 360//2
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

# Grep from Storage 
if exists("mask_storage"):
    storage = shelve.open("mask_storage")
    low_H = storage['low_H']
    low_S = storage['low_S']
    low_V = storage['low_V']
    high_H = storage['high_H']
    high_S = storage['high_S']
    high_V = storage['high_V']
    storage.close()
else:
    low_H = 0
    low_S = 0
    low_V = 10
    high_H = 175
    high_S = 100
    high_V = 255

if exists("mask_storage_homo"):
    storage_homo = shelve.open("mask_storage_homo")
    low_H_homo = storage_homo['low_H_homo']
    low_S_homo = storage_homo['low_S_homo']
    low_V_homo = storage_homo['low_V_homo']
    high_H_homo = storage_homo['high_H_homo']
    high_S_homo = storage_homo['high_S_homo']
    high_V_homo = storage_homo['high_V_homo']
    storage_homo.close()
else:
    low_H_homo = 40
    low_S_homo = 60
    low_V_homo = 60
    high_H_homo = 86
    high_S_homo = 230
    high_V_homo = 230

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_low_H_thresh_trackbar_homo(val):
    global low_H_homo
    global high_H_homo
    low_H_homo = val
    low_H_homo = min(high_H_homo-1, low_H_homo)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H_homo)

def on_high_H_thresh_trackbar_homo(val):
    global low_H_homo
    global high_H_homo
    high_H_homo = val
    high_H_homo = max(high_H_homo, low_H_homo+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H_homo)

def on_low_S_thresh_trackbar_homo(val):
    global low_S_homo
    global high_S_homo
    low_S_homo = val
    low_S_homo = min(high_S_homo-1, low_S_homo)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S_homo)

def on_high_S_thresh_trackbar_homo(val):
    global low_S_homo
    global high_S_homo
    high_S_homo = val
    high_S_homo = max(high_S_homo, low_S_homo+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S_homo)

def on_low_V_thresh_trackbar_homo(val):
    global low_V_homo
    global high_V_homo
    low_V_homo = val
    low_V_homo = min(high_V_homo-1, low_V_homo)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V_homo)

def on_high_V_thresh_trackbar_homo(val):
    global low_V_homo
    global high_V_homo
    high_V_homo = val
    high_V_homo = max(high_V_homo, low_V_homo+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V_homo)

def live_seperation_homo(img : list):

    if not exists("mask_storage_homo"):
        # Resize image
        height, width, dim = img.shape
        img = cv.resize(img.copy(),(int(width/4), int(height/4)) , interpolation= cv.INTER_LINEAR)

        cv.namedWindow(window_capture_name)
        cv.namedWindow(window_detection_name)
        cv.createTrackbar(low_H_name, window_detection_name , low_H_homo, max_value_H, on_low_H_thresh_trackbar_homo)
        cv.createTrackbar(high_H_name, window_detection_name , high_H_homo, max_value_H, on_high_H_thresh_trackbar_homo)
        cv.createTrackbar(low_S_name, window_detection_name , low_S_homo, max_value, on_low_S_thresh_trackbar_homo)
        cv.createTrackbar(high_S_name, window_detection_name , high_S_homo, max_value, on_high_S_thresh_trackbar_homo)
        cv.createTrackbar(low_V_name, window_detection_name , low_V_homo, max_value, on_low_V_thresh_trackbar_homo)
        cv.createTrackbar(high_V_name, window_detection_name , high_V_homo, max_value, on_high_V_thresh_trackbar_homo)

        while True:
            if img is None:
                break
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H_homo, low_S_homo, low_V_homo), (high_H_homo, high_S_homo, high_V_homo))
            
            cv.imshow(window_capture_name, img)
            cv.imshow(window_detection_name, frame_threshold)
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break

        cv.destroyAllWindows()

        storage_homo = shelve.open("mask_storage_homo")
        storage_homo['low_H_homo'] = low_H_homo
        storage_homo['low_S_homo'] = low_S_homo
        storage_homo['low_V_homo'] = low_V_homo
        storage_homo['high_H_homo'] = high_H_homo
        storage_homo['high_S_homo'] = high_S_homo
        storage_homo['high_V_homo'] = high_V_homo
        storage_homo.close()

    lower = np.array([low_H_homo, low_S_homo, low_V_homo])
    higher = np.array([high_H_homo, high_S_homo, high_V_homo])

    return lower, higher

def live_seperation(img : list):
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

    while True:
        if img is None:
            break
        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        
        cv.imshow(window_capture_name, img)
        cv.imshow(window_detection_name, frame_threshold)
        
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break

    cv.destroyAllWindows()
    storage = shelve.open("mask_storage")
    storage['low_H'] = low_H
    storage['low_S'] = low_S
    storage['low_V'] = low_V
    storage['high_H'] = high_H
    storage['high_S'] = high_S
    storage['high_V'] = high_V
    storage.close()

    lower = np.array([low_H, low_S, low_V])
    higher = np.array([high_H, high_S, high_V])

    return lower, higher
