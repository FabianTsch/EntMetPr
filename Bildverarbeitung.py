# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:19:54 2022

@author: Stefan Kaufmann
Abschlussprojekt

1_ Bildbearbeitung
"""

#standard import
from email.headerregistry import AddressHeader
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Open CV
import cv2
from math import atan2, cos, sin, sqrt, pi


class Bilder:  
    
    x = 750                                  # Width of the img after homography
    y = 500                                  # Hight of the img after homography
    j = np.array([], dtype=int)              # Counturs of interest after Counturs filter
    count = 0                                # to devide the two mask types (before and after homography)
    obj = []                                 # np.array([x,y,w,h])  dimensions of the rec_counturs
    im_dst = 0                               # img after Homography as RGB
    plot = True                              # Plotten der Schwereachsen


       
    def __init__(self, img):              
         self.img = img                     # origin img 
         
    def maske(self):
        """Creats a Mask for the Origin img and for the img after the homography
            Params
             --------
                  plot:   Plots both masks 
                  count:  to devide the two mask types (before and after homography)

            Returns
            --------
                mask:  mask as binary img
        """
        
        if self.count == 0:
            lower = np.array([40,80,80]) 
            upper = np.array([86,230,230]) 
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            self.mask = cv2.inRange(hsv, lower, upper) 
            if self.plot:
                cv2.imshow('Maske', self.mask)
            
        else:
            gray = cv2.cvtColor(self.im_dst, cv2.COLOR_BGR2GRAY)
            self.mask = cv2.inRange(gray, 130, 255)
            self.j = []                    # Clear the storage
            if self.plot:
                cv2.imshow('Maske2', self.mask)
           
       

      
    
    def homography(self):    

        """Call  maske() and segmentation(), sort the Points and make the homography
            Params
             --------
                                   

            Returns
            --------
                im_dst:  img after homography as RGB
        """
        
        if self.plot:
            print('aurichten')
         
         # find Objects  
        self.maske()
        self.segmentation()
        
        
        # MC sortieren                    
        mc = self.mc
        self.mc = self.mc[self.mc[:,1].argsort()]        
        #print(self.mc)
        if self.mc[0,0] < self.mc[1,0]:            
            temp = np.copy(self.mc[0,:])
            self.mc[0,:] = self.mc[1,:]
            self.mc[1,:] = temp
    
        if self.mc[2,0] > self.mc[3,0]:             
              temp = np.copy(self.mc[2,:])
              self.mc[2,:] = self.mc[3,:]
              self.mc[3,:] = temp
        
                        
        # Points in destination image        
        points_dst = np.array([ [self.x, 0], [0, 0],[0, self.y],[self.x, self.y] ])
        
        # Homography
        h, status = cv2.findHomography(self.mc, points_dst)          
        self.im_dst = cv2.warpPerspective(self.img, h, (self.x,self.y))
        if self.plot:
            print('Ende erster Durchlauf')
    
        
    def segmentation(self):
        """Finds the Counturs, center of mass, main direction and Filters Conturs that are not possible 
            Params
             --------                                 

            Returns
            --------
                
        """

        self.contours, hierarchy = cv2.findContours(self.mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
        # Counturs Filtern 
        for i in range(len(self.contours)):       
            area = cv2.contourArea(self.contours[i])
            if area > 150 and self.count == 0:                 
                self.j = np.append(self.j,[i] )
            elif area > 100  and self.count > 0:                   # Fläche Überprüfen
                x,y,w,h = cv2.boundingRect(self.contours[i]) 
                if 0< x < self.x and 0< y < self.y:                # Position Überprüfen
                    obj =      np.array([x,y,w,h]) 
                    self.j =   np.append(self.j,[i] )  
                    self.obj = np.append(self.obj,obj)

        self.j = self.j[0:len(self.j)]  
        self.j = self.j.astype(int)
        
        if self.count > 0:
            self.obj = np.resize(self.obj,(len(self.j),4))   
            self.obj = self.obj.astype(int)


        # Get the moments
        mu = [None]*len(self.j)
        for i in range(len(self.j)):            
            mu[i] = cv2.moments(self.contours[self.j[i]])


        # Get the mass centers
        mc = [None]*len(self.j)
        for i in range(len(self.j)):           
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))        
        mc = np.asarray(mc)   # Konvertierung in ein Array        
        self.mc = mc.astype(int)
        
        
        if self.count > 0:
                   
            # Get the orientation
            mo = [None]*len(self.j)
            for i in range(len(self.j)):   
                mo[i] = self.getOrientation(mu[i])
            self.mo = np.asarray(mo)
            
            # Get mini Pictures of each obj            
            mp = [None]*len(self.j)
            for i in range(len(self.j)):   
                mp[i] = self.cut(self.im_dst, self.obj[i])            
            self.mp = mp

        self.count += 1 # Nächster Schritt
    

    def getOrientation(self, mu):
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
        
        alpha = np.round(atan2(ev[0,1],ev[1,1])*180/pi-90 +360   ,1)           
    
        s = 20  # Skalierung der Pfeile
        
        if self.plot == True:            
            font = cv2.FONT_HERSHEY_SIMPLEX   
            fontScale = 0.6
            color = (0, 0, 0)
            thickness = 1
            cv2.putText(self.im_dst, str(alpha), (x,y) , font, fontScale, color, thickness)
            image = cv2.line(self.im_dst, center, (x+int(ev[0,0]*s),y+int(ev[1,0]*s)), (0,255,0), 2)
            self.image = cv2.line(self.im_dst, center, (x+int(ev[0,1]*s*3),y+int(ev[1,1]*s*3)), (0,0,0), 2)                     
            cv2.imshow('Schwereachsen', self.image)        
         
        return alpha
        
     
    def cut(self, img, obj):

        """ Cut the given img around the given objectsize and the center of mass with some overhang
            Params
            --------
             img: picture as RGB  
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
        if self.plot == True:
            cv2.imshow('Bilder_zugeschnitten', image)           
        return image

    def findObject(self):

        """ Makes a mask and find the Object after the Homography 
            --------                                  

            Returns
            --------
                            
        """
        self.maske()
        self.segmentation()


    def display_img(self):
        """ Shows some pictures of the process
                                    
        """


        if self.plot:     
                       

            if self.count>0:
                
                a = 10
                for i in range(len(self.j)):

                    cv2.rectangle(
                        self.im_dst,
                        (self.obj[i, 0] - a, self.obj[i, 1] - a),
                        (self.obj[i, 0] + self.obj[i, 2] + a, a + self.obj[i, 1] + self.obj[i, 3]),
                        (0, 255, 0),
                        1,
                    )
                    cv2.imshow("im_dst mit Schwereachsen", self.im_dst)

 
def execute():
    # TODO: This is the main function of image processing
    pass
