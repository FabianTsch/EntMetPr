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


def calc_orientation(mu):
    """Finds the main Orientation of the given Contour
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
    for i in range(len(contours)):       
        area = cv2.contourArea(contours[i])

        if area > 100:                   # Fläche Überprüfen
            x,y,w,h = cv2.boundingRect(contours[i]) 
            if 0< x < IMAGE_WIDTH and 0< y < IMAGE_HIGHT:                # Position Überprüfen
                obj_buffer = np.array([x,y,w,h]) 
                contour_points = np.append(contour_points,[i] )  
                obj = np.append(obj,obj_buffer)

    contour_points = contour_points[0:len(contour_points)]  
    contour_points = contour_points.astype(int)
    
    obj = np.resize(obj,(len(contour_points),4))   
    obj = obj.astype(int)

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
        mo[i] = calc_orientation(mu[i])
    mo = np.asarray(mo)

    # Get mini Pictures of each obj
    mp = [None]*len(contour_points)
    for i in range(len(contour_points)):
        mp[i] = crop_image(img, obj[i])

    # change color BGR to RGB
    img_array = [None]*len(mp)
    for i in range(0,len(mp)):
            img_array[i] = cv2.cvtColor(mp[i],cv2.COLOR_BGR2RGB)  

    return img_array, obj[:,0], obj[:,1], mo
