import matplotlib as plt
import numpy as np
from math import pi
from email.headerregistry import AddressHeader

# Open CV
import cv2
from math import atan2, cos, sin, sqrt, pi

TARGET_HOMOGRAPHY_POINTS = 0
TARGET_OBJECTS = 1 
IMAGE_WIDTH = 750
IMAGE_HIGHT = 500


def calc_angle(mu):
    """ calc angle through given moment
        Params
         --------
         mu:   Moments of the Contour                                 

        Returns
        --------
        apha: Main Dircetion of the contour
            
    """

    x = int(mu["m10"] / mu["m00"])
    y = int(mu["m01"] / mu["m00"])
    center = (x,y)
    
    a = int(mu["m20"] - mu["m10"]*mu["m10"]/mu["m00"])
    b = int(mu["m02"] - mu["m01"]*mu["m01"]/mu["m00"])
    c = int(mu["m11"] - mu["m10"]*mu["m01"]/mu["m00"])
    
    
    J = np.array([[a, c],[c, b]])
    ew,ev = np.linalg.eig(J)
    
    alpha = np.round(atan2(ev[0,1],ev[1,1])*180/pi +360, 1)           

    return alpha

def plot_contour(img,contour):
    """ debug function to plot contours 
        Params
        --------
         img: 3D RGB array
         contour: contour for plotting

        Returns
        --------
    """

    image_contour = cv2.drawContours(img,contour,-1,(0,255,0),1)
    cv2.imshow("singel contour",image_contour)
    cv2.waitKey(0)

def calc_aoi(a,b,c,d):
    """ calculates area of intersection
        Params
        --------
         a,b: corners of the first rectangle 
         c,d: corners of the second rectangle
        Returns
        --------
         aoi: area of intersection 
    """
    x, y = 0, 1
    width = min(b[x], d[x]) - max(a[x],c[x])
    height = min(b[y],d[y]) - max(a[y],c[y])

    if min(width, height) > 0: 
        return width * height
    else: 
        return 0


def find_objects(contoursg,img):
    """ finds object in given contour and detects
        if the obj is standing of lying
        Params
        --------
         contours: contours found in the img
         img: just for debugging
        Returns
        --------
         obj: found objects
         orientation: standing or lying
         contour_buffer: contour corresponding to the obj
    """
    obj = []
    orientation = []
    contour_buffer = []
    boundRect = [None]*len(contours)
    area_min = 100
    valid_area = (621,500)

    # condition lying
    area_lying = (1100,4000)
    area_overlapping_threshold = 0.9

    # get relevant contours
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i]) 
        if area > area_min and 0< x < valid_area[0] and 0< y < valid_area[1]:
            contour_buffer.append(contours[i])

    contour = contour_buffer
    contour_buffer = []

    # check standing or lying
    for i in range(0,len(contour)):
        x,y,w,h = cv2.boundingRect(contour[i])
        boundRect = cv2.boundingRect(contour[i])
        x2,y2,w2,h2 = cv2.boundingRect(contour[i-1])
        area = h * w
        a = [x,y]
        b = [x+w,y+h]
        c = [x2,y2]
        d = [x2+w2,y2+h2]
        aoi = calc_aoi(a,b,c,d)

        if area_lying[0] < area < area_lying[1]:
            obj.append(np.array([x,y,w,h]))
            orientation.append(2)
            contour_buffer.append(contour[i])
        elif aoi / area > area_overlapping_threshold:

            if w*h <= w2*h2:
                obj.append(np.array([x,y,w,h]))
                contour_buffer.append(contour[i])
                orientation.pop(-1)
                obj.pop(-1)
                contour_buffer.pop(-1)
            else:
                pass
            # TODO: Scenario abdecken, wenn inner Mutterfläche 
            # vor äußeren auftritt (scheint nie vorzukommen)

            orientation.append(2)
        else:
            obj.append(np.array([x,y,w,h]))
            contour_buffer.append(contour[i])
            orientation.append(1)
    return obj, orientation, contour_buffer


def crop_image(img, obj):
    """ Cut the given img around the given objectsize and the center of mass with some overhang
        Params
        --------
         img: picture as BGR  
         obj: array with the object size [x,y,w,h]                               

        Returns
        --------
        image: the cut img with some overhang d
            
    """
    d = 2                        # image overhang
    x = int(obj[0])
    y = int(obj[1])
    w = int(obj[2])
    h = int(obj[3])
           
    image = img[(y-d):(y+h+d), (x-d):(x+w+d)]
    return image

def create_mask(img, target):
    if target == TARGET_HOMOGRAPHY_POINTS:
        lower = np.array([40,80,80]) 
        upper = np.array([86,230,230]) 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, lower, upper) 
        
    elif target == TARGET_OBJECTS:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.inRange(gray, 130, 255)


def homography(img):
    mask = create_mask(img,target=TARGET_HOMOGRAPHY_POINTS)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
    # Counturs Filtern 
    contour_points = np.array([], dtype=int)
    for i in range(len(contours)):       
        area = cv2.contourArea(contours[i])

        if area > 150:                 
            contour_points = np.append(contour_points,[i] )

    contour_points = contour_points[0:len(contour_points)]  
    contour_points = contour_points.astype(int)

    # Get the moments
    mu = [None]*len(contour_points)
    for i in range(len(contour_points)):            
        mu[i] = cv2.moments(contours[contour_points[i]])

    # Get the mass centers
    mc = [None]*len(contour_points)
    for i in range(len(contour_points)):           
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))        
    mc = np.asarray(mc)   # Konvertierung in ein Array        
    mc = mc.astype(int)

    # sort the mass center points                    
    mc = mc[mc[:,1].argsort()]        
    if mc[0,0] < mc[1,0]:            
        temp = np.copy(mc[0,:])
        mc[0,:] = mc[1,:]
        mc[1,:] = temp

    if mc[2,0] > mc[3,0]:             
          temp = np.copy(mc[2,:])
          mc[2,:] = mc[3,:]
          mc[3,:] = temp
    
    # Points in destination image        
    points_dst = np.array([ [IMAGE_WIDTH, 0], [0, 0],[0, IMAGE_HIGHT],[IMAGE_WIDTH, IMAGE_HIGHT] ])
    
    # Homography
    h, status = cv2.findHomography(mc, points_dst)          
    image = cv2.warpPerspective(img, h, (IMAGE_WIDTH,IMAGE_HIGHT))
    return image


def object_detection(img):
    mask = create_mask(img,target=TARGET_OBJECTS)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contour filter
    contour_points = np.array([], dtype=int)
    obj = []
    obj, orientation, contours = find_objects(contours,img)
    contour_points = range(0,len(contours))

    # Get the moments
    mu = [None]*len(contour_points)
    for i in range(len(contour_points)):            
        mu[i] = cv2.moments(contours[contour_points[i]])

    # Get the mass centers
    mc = [None]*len(contour_points)
    for i in range(len(contour_points)):           
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))        
    mc = np.asarray(mc)   # Konvertierung in ein Array        
    mc = mc.astype(int)
    
    # Get the orientation
    mo = [None]*len(contour_points)
    for i in range(len(contour_points)):   
        mo[i] = calc_angle(mu[i])
    mo = np.asarray(mo)

    # Get mini Pictures of each obj
    mp = [None]*len(contour_points)
    for i in range(len(contour_points)):
        mp[i] = crop_image(img, obj[i])

    # change color BGR to RGB
    img_array = [None]*len(mp)
    for i in range(0,len(mp)):
            img_array[i] = cv2.cvtColor(mp[i],cv2.COLOR_BGR2RGB)  

    # Make List of Array to 2D Array
    obj = np.vstack(obj)
    return img_array, obj[:,0], obj[:,1], mo, orientation
