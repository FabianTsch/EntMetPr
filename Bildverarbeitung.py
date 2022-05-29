import matplotlib as plt
import numpy as np
from math import pi
from email.headerregistry import AddressHeader

# Open CV
import cv2
from math import atan2, cos, sin, sqrt, pi


class Bilder:  
    def __init__(self, img):              
        self.__TARGET_HOMOGRAPHY_POINTS = 0
        self.__TARGET_OBJECTS = 1 

        self.__img = img                                # origin img 
        self.__x = 750                                  # Width of the img after homography
        self.__y = 500                                  # Hight of the img after homography
        self.__contours = np.array([], dtype=int)       # contours of interest after Counturs filter
        self.__count = 0                                # to devide the two mask types (before and after homography)
        self.__obj = []                                 # np.array([x,y,w,h])  dimensions of the rec_counturs
        self.__mc = []                                  # mass center points
        self.__im_dst = 0                               # img after Homography as RGB
        self.__mp = [None]                              # array for mini pictures
        self.plot = False                             # Plotten der Schwereachsen

        self.find_objects(target=self.__TARGET_HOMOGRAPHY_POINTS)
        self.homography()
        self.find_objects(target=self.__TARGET_OBJECTS)

    @property
    def mp(self):
        return self.__mp

    def maske(self, target):
        """Creats a Mask for the Origin img and for the img after the homography
            Params
             --------
                  plot:   Plots both masks 
                  count:  to devide the two mask types (before and after homography)

            Returns
            --------
                mask:  mask as binary img
        """
        
        if target == self.__TARGET_HOMOGRAPHY_POINTS:
            lower = np.array([40,80,80]) 
            upper = np.array([86,230,230]) 
            hsv = cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, lower, upper) 
            
        elif target == self.__TARGET_OBJECTS:
            gray = cv2.cvtColor(self.__im_dst, cv2.COLOR_BGR2GRAY)
            self.__contours = [] # Clear the storage
            return cv2.inRange(gray, 130, 255)
           

    def homography(self):    
        """Call  maske() and segmentation(), sort the Points and make the homography
            Params
             --------
                                   
            Returns
            --------
                im_dst:  img after homography as RGB
        """
        # MC sortieren                    
        self.__mc = self.__mc[self.__mc[:,1].argsort()]        
        #print(self.mc)
        if self.__mc[0,0] < self.__mc[1,0]:            
            temp = np.copy(self.__mc[0,:])
            self.__mc[0,:] = self.__mc[1,:]
            self.__mc[1,:] = temp
    
        if self.__mc[2,0] > self.__mc[3,0]:             
              temp = np.copy(self.__mc[2,:])
              self.__mc[2,:] = self.__mc[3,:]
              self.__mc[3,:] = temp
        
                        
        # Points in destination image        
        points_dst = np.array([ [self.__x, 0], [0, 0],[0, self.__y],[self.__x, self.__y] ])
        
        # Homography
        h, status = cv2.findHomography(self.__mc, points_dst)          
        self.__im_dst = cv2.warpPerspective(self.__img, h, (self.__x,self.__y))
        if self.plot:
            print('Ende erster Durchlauf')
    
        
    def segmentation(self,mask,target):
        """Finds the Counturs, center of mass, main direction and Filters Conturs that are not possible 
            Params
             --------                                 

            Returns
            --------
                
        """

        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
        # Counturs Filtern 
        for i in range(len(contours)):       
            area = cv2.contourArea(contours[i])

            if area > 150 and target == self.__TARGET_HOMOGRAPHY_POINTS:                 
                self.__contours = np.append(self.__contours,[i] )

            elif area > 100  and target == self.__TARGET_OBJECTS:                   # Fläche Überprüfen
                x,y,w,h = cv2.boundingRect(contours[i]) 
                if 0< x < self.__x and 0< y < self.__y:                # Position Überprüfen
                    obj =      np.array([x,y,w,h]) 
                    self.__contours =   np.append(self.__contours,[i] )  
                    self.__obj = np.append(self.__obj,obj)

        self.__contours = self.__contours[0:len(self.__contours)]  
        self.__contours = self.__contours.astype(int)
        
        if target == self.__TARGET_OBJECTS:
            self.__obj = np.resize(self.__obj,(len(self.__contours),4))   
            self.__obj = self.__obj.astype(int)


        # Get the moments
        mu = [None]*len(self.__contours)
        for i in range(len(self.__contours)):            
            mu[i] = cv2.moments(contours[self.__contours[i]])


        # Get the mass centers
        mc = [None]*len(self.__contours)
        for i in range(len(self.__contours)):           
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))        
        mc = np.asarray(mc)   # Konvertierung in ein Array        
        self.__mc = mc.astype(int)
        
        
        if target == self.__TARGET_OBJECTS:
                   
            # Get the orientation
            mo = [None]*len(self.__contours)
            for i in range(len(self.__contours)):   
                mo[i] = self.getOrientation(mu[i])
            self.mo = np.asarray(mo)

            # Get mini Pictures of each obj
            self.__mp = [None]*len(self.__contours)
            for i in range(len(self.__contours)):
                self.__mp[i] = self.cut(self.__im_dst, self.__obj[i])


    def find_objects(self,target):
        maske = self.maske(target)
        self.segmentation(maske,target)
    

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
        if self.plot:
            cv2.imshow('Bilder_zugeschnitten', image)           
        return image

def resize(img, size=[224,224], color=[55, 74, 195]):
    """ resizes image to size and fills up emty space with color
    Params
    --------
    img: picture as RGB
    size: desired size of the cuttted image
    color: color to fill up empty space

    Returns
    --------
    image: resized image
    """
    height, width = img.shape[:2]
    x_offset = int(size[0]/2)-int(width/2) 
    y_offset = int(size[1]/2)-int(height/2)
    size = np.append(size,3) 

    blank_image = np.zeros(size, np.uint8)
    blank_image[:, :] = color 

    image = blank_image.copy()

    image[y_offset:y_offset+height, x_offset:x_offset+width] = img.copy()

    return image


def execute(img):
    """ searches for object in img and extracts them into
    img_arrry
    Params
    --------
    img: Image including all Objects 

    Returns
    --------
    image: Image array with seperated objects 
    """
    b = Bilder(img)

    # change color BGR to RGB
    img_array = [None]*len(b.mp)
    for i in range(0,len(b.mp)):
            img_array[i] = cv2.cvtColor(b.mp[i],cv2.COLOR_BGR2RGB)  

    return img_array
