"""
Created on Fr May  20 15:43:00 2022

@author: Stefan Kaufmann
Abschlussprojekt

Communication: Filters the array of object checks for collisions 
"""

# %%
from turtle import width
import cv2
import numpy as np
from torch import threshold

plot = True



def createObj(type, orientation,x,y,angle, orig, tool=False,move=False, workspace=100):
    """Creats a obj, screw nut, lying/standing 
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image   
        tool:          if we want to grab with tool   # true = tool, false = no tool    default False 
        move:          if we want to move the obj with a tool, # true = move
        workspace:     work space is the overlay of the table, we need it for the implementation default 100 mm
        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no obj            
                
    """
    
    # Creat Workspace and Table
    
    img = np.zeros((orig[1]+2*workspace, orig[0]+2*workspace))
    if plot:
        img[workspace,:] = 1
        img[workspace+orig[1],:] = 1
        img[:,workspace] = 1
        img[:,workspace+orig[0]] = 1
        
    # Security Space
    s = 5 # [mm]

    #  declared the opbject space as an rect
    if tool:
        if type == 1 and orientation == 2:  # tool for screw 
            width = 42+s
            height = 20+s
           
                        
        else:   # tool for nuts
            width = 42+s
            height = 20+s     
   
    elif type == 1 and orientation == 2:   # screws
        width = 20
        height = 50
    else:                                  # nuts
        width = 20
        height = 20
    
    if move:       
        width = 42+s
        height = 20+s

    obj = np.ones((height,width))
    if tool and not move:
        obj[:,2*s:width-2*s] = 0

    

        
    
    # Rotate the img
    w = int(np.sqrt(np.square(height)+np.square(width)))                        # new size of the img

    M = np.float32([[1,0,int((w-width)/2)],[0,1,int((w-height)/2)]])            # translation to the new center
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper verschoben', obj)
      
    rows,cols = obj.shape
    M = cv2.getRotationMatrix2D((w/2,w/2), angle, 1)
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper gedreht', obj)
    

    # Draw the obj in the work space 
    rows,cols = obj.shape   
    
    for i in range(0,rows):
        for j in range(0,cols):
            if obj[i,j] == 1 or img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] == 1:        # if obj or img at the location of the obj is 1 --> img at the location is 1
                img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] = 1                 

    if plot:
         return img 
          
    return img[workspace:orig[1]+workspace, workspace:orig[0]+workspace]

def createImg(type, orientation,x,y,angle, orig,tool=False,move = False, workspace=100):
    """Creats a obj, screw nut, lying/standing, can handle scalars and arrays
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        move:          if we want to move the obj with a tool, # true = move
        workspace:     work space is the overlay of the table, we need it for the implementation,  default 100 mm

        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no obj            
                
    """
    if plot:
        img = np.zeros((orig[1]+2*workspace, orig[0]+2*workspace))
    else:
        img = np.zeros((orig[1], orig[0]))

    if np.isscalar(type):
        img = createObj(type,orientation,x,y,angle, orig, tool,move, workspace)
    else:
        for i in range(0,np.size(type)):
            img += createObj(type[i],orientation[i],x[i],y[i],angle[i], orig, tool,move, workspace)           
    
    return img

def pick(type, orientation,x,y,angle,img,move=False, workspace=100):
    """ Checks if the obj can be picked
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        img:           Binary image with all remaining Objects 1 = object  
        move:          if we want to move the obj with a tool, # true = move
        tool:          if we want to grab with tool   # true = tool, false = no tool                    

        Returns
        --------
        posible:       True if the obj can be picked           
                
    """
    # creats a tool img
    y_img,x_img = img.shape
    
    if plot:
        y_img -= workspace*2
        x_img -= workspace*2
    img_tool = createObj(type, orientation,x,y,angle, (x_img,y_img), True, move)

    img_bitwise_and = cv2.bitwise_and(img, img_tool)
    if img_bitwise_and.any():
        possible = False
    else:
        possible = True

    if plot:
        img_bitwise_and = img_bitwise_and[(workspace+1):orig[1]+workspace, (workspace+1):orig[0]+workspace]  # shrinks the image to the area of ​​interest
        #cv2.imshow('bitwise_and', img_bitwise_and)
        #cv2.waitKey()
        if img_bitwise_and.any():
             possible = False
        else:
            possible = True
        print(' ISt das Objekt audgreifbar?', possible)

    return possible

def checkPick(type, orientation,x,y,angle,orig, img = 0):
    """ Checks if the obj can be picked works with scalars and arrays
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        img:           Binary image with all remaining Objects 1 = object, by default creats his own binary
        tool:          if we want to grab with tool   # true = tool, false = no tool                    

        Returns
        --------
        j:             Array of indixes of the obj which can be picked in the correct order  
        posible:       True if the obj can be picked       
                
    """
    # Create reconstructed image
    if img == 0:
        img = createImg(type, orientation,x,y,angle, orig)       
       
    if plot:
        cv2.imshow('Das Orginal', img) 
    # Helpful variables
    possible = []
    j = []                  # Array of indixes of the obj which can be picked in the correct order

    if np.isscalar(type): 
        possible = pick(type, orientation,x,y,angle,img)
        return possible
    
    
    else:
        for count in range(0,np.size(type)):  # do the check as often as there are elements in obj  
            if plot:
                print ('#################  Durlauf Nr.: ', count)            

            for i in range(0,np.size(type)):  # first loop over all objects  
                if count%2 == 0:
                    i = np.size(type)-i -1
                
                if plot:
                    print('Der Sublauf: ', i)   
                    print('##### j ist: ', j)          
                if not i in j and pick(type[i], orientation[i],x[i],y[i],angle[i],img):   # Controls if any i ist part of j and if obj[i] is pickable             
                                   
                    j = np.append(j,i)   
                    img -=createImg(type[i], orientation[i],x[i],y[i],angle[i],orig) # subtracts the object from the image  
                    if plot:                                             
                        cv2.imshow(str(i), img)  
                        cv2.waitKey()            
                    
            

    return j

def nearObj(x,y,x1,y1):
    """Creats a Group of obj with are neare 
        Params
         --------
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position   
        threshold:     the minimal distance between the obj 
       
        Returns
        --------
        near:          True if the Object is near               
                
    """
    near = False
    
    # dim of scew and nut
    width_s = 50
    height_s = 20
    nut = 20
    
    # threshold = worst case, screw lies 45° at the center of the nuts edge
    threshold = ((width_s**2 + height_s**2)**0.5 + nut)/2
    # amount of the vector
    distance = ((x-x1)**2+(y-y1)**2)**0.5

    if distance < threshold:
     near = True

    return near

def createSubGroup(type, orientation,x,y,angle, area=0):
    """Creats a Group of obj with are neare
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad    
        area:          area of the given objects in mm²   
       
        Returns
        --------
        g_x:           x Position of the Group in mm from the origin
        g_y:           y Position of the Group in mm from the origin
        g_angle:       Orientation of the main axis in grad    
        g_area:        area of the group in mm²            
                
    """
    g_x = []
    g_y = []
    g_angle = [] 
    g_area = []
    j = []            # inidizes of whitch obj that are not seperated
    g_type = np.zeros(np.size(x))     # helpful variable to declare if an obj is part of a Group
    
    # Error handling if we get an scalar instead of an list    
    if  np.size(x) == 1:
        return x, y, angle, area
         
    if np.isscalar(area):                         # For the first loop
        surface = np.ones(np.size(type))   

        
        for i in range(0,np.size(type)):
            if type[i] == 1 and orientation[i] == 2:
                surface[i] = 2
            elif type[i] == 2 and orientation[i] == 1:
                surface[i] == 0.5
        print('Areas',surface) 
        print('orientation',orientation) 
    else:
        surface = area
    

    # creat group 
       
    for i in range(1,np.size(x)):
        if nearObj(x[i-1], y[i-1], x[i],y[i]) and not g_type[i-1] == 3:
            # common parameters based on their areas 
            g_x = np.append(g_x,np.abs((x[i-1]*surface[i-1]+x[i]*surface[i])/(surface[i-1]+surface[i])))
            g_y = np.append(g_y,np.abs((y[i-1]*surface[i-1]+y[i]*surface[i])/(surface[i-1]+surface[i])))
            g_angle = np.append(g_angle,np.abs((angle[i-1]*surface[i-1]+angle[i]*surface[i])/(surface[i-1]+surface[i])))
            g_area = np.append(g_area,surface[i-1]+surface[i] )
            g_type[i] = 3 
            g_type[i-1] = 3 
            if plot:
              print('Objekt erstellen')
        else:
            j = np.append(j,int(i))
               
    # attach  
    for i in j:
        i = int(i)
        g_x = np.append(g_x, x[i] )  
        g_y = np.append(g_y, y[i] ) 
        g_angle = np.append(g_angle, angle[i]) 
        g_area = np.append(g_area, surface[i])

    
    

    
    print('Ende creatGroup')
    return g_x, g_y, g_angle, g_area

def createGroup(type, orientation,x,y,angle,j):
    """Creats a Group of obj with are neare
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad    
        area:          area of the given objects in mm²   
       
        Returns
        --------
        m_type:        screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        m_orientation: standing = 1, lying = 2,  perhabs more info  
        m_x:           x Position of the Group in mm from the origin
        m_y:           y Position of the Group in mm from the origin
        m_angle:       Orientation of the main axis in grad    
                
                
    """        
    m_type = []
    m_orientation = []
    m_x = []
    m_y = []
    m_angle = []
    

    # creat a list with obj that can not be picked
    for i in range(0,np.size(type)):

        if not i in j:
            m_type = np.append(m_type, type[i])
            m_orientation = np.append(m_orientation, orientation[i])
            m_x = np.append(m_x, x[i])
            m_y = np.append(m_y, y[i])
            m_angle = np.append(m_angle, angle[i])   
                  
    

    # first loop
    m_x, m_y, m_angle, m_area = createSubGroup(m_type,m_orientation,m_x,m_y,m_angle)    
    # other loops
    for i in range(0,np.size(type)):
        print('####################### Runde :', i)
        m_x, m_y, m_angle, m_area = createSubGroup(m_type,m_orientation,m_x,m_y,m_angle,m_area)
    
    
    
    
    m_orientation = np.ones(np.size(m_x)) *2    # Orientation chanced to lying
    m_type = np.ones(np.size(m_x)) *3            # Type chanced to seperate
    print('Area:', m_area)
    print('Ende Fkt')
    return m_type, m_orientation, m_x, m_y, m_angle
 
def move(type, orientation,x, y, angle, orig):
    """Creats the attack angle
        Params
         --------
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad   
        
       
        Returns
        --------
        
        m_angle:       Orientation of the attack angle of the group  
        m_type:        set type seperate    
        error:         if there is error, 1 = no rechable Positionen found          
                
    """  
   

    # check if their are more than one group
    if np.size(x) == 1:
        return angle + 90

    # important variables
    threshold = np.inf
    alpha = np.zeros(np.size(x))
    error = np.zeros(np.size(x))

    # creat img with all groups
    img = createImg(type, orientation,x,y,angle, orig)

    
    
    # Corrects unwanted angles
    for i in range (0,np.size(x)):
        if angle[i] < 0:
            angle[i] += 360

        if angle[i] > 180:
            angle[i] -=180

    # find nearts group    
    for count in range (0,np.size(x)):        
        r = 0
        z = 0
        # first step, check for the nearest group
        for i in range(0,np.size(x)):
            if not count == i:    
                r = ((x[count]-x[i])**2+(y[count]-y[i])**2)**0.5             
                if r < threshold:
                    j = i   # nearest group
        
        # second step search for the ideal attack vector
        # creat the vektor between the two Groups
        m_alpha = np.arctan2((y[j]-y[count]),(x[j]-x[count]))  

        # check if this position is reachable      
        x_m = r*np.cos(m_alpha)/2 + x[count]
        y_m = r*np.sin(m_alpha)/2 + y[count] 
        while not pick(1,2,x_m,y_m,m_alpha,img,True):  # if alpha is not poissible --> rotate alpha 
            m_alpha += 5    
            z += 5
            if z > 360:
                print('Error --> keine erreichbare Position gefunden')
                error[count] = 1
                break 
           
        alpha[count] = m_alpha*180/np.pi+180
 
    m_type = np.ones(np.size(x)) *3           # set type seperate

    return alpha, m_type, error

  

# %%

# ****************************************************************************************************************************************************************
# main
# ****************************************************************************************************************************************************************
#For Testing
# image size
x = 750          
y = 500

type = [1,1,1,1]
orientation = [1,2, 1, 2]
x_obj = [25,200, 180, 180]
y_obj = [25,200, 400, 200]
angle = [0,0, 0, 20]
orig = [x,y]

img = createImg(type, orientation,x_obj,y_obj,angle, orig)
img_tool =  createImg(type, orientation,x_obj,y_obj,angle, orig, True )



# %%
#First Part --> seperating all objetcs who can be picked
print('##############################################################    Erster Teil')
j = checkPick(type, orientation,x_obj,y_obj,angle,orig)
print('Folgende Indizies sind aufgreifbar: ', j)
cv2.imshow('alle Objekt', img+img_tool)
img_tool =  np.zeros((orig[1], orig[0]))
img = np.zeros((orig[1], orig[0]))
for i in j:
    i = int(i)
    img += createImg(type[i], orientation[i],x_obj[i],y_obj[i],angle[i], orig )
    img_tool += createImg(type[i], orientation[i],x_obj[i],y_obj[i],angle[i], orig, True )    
cv2.imshow('alle Objekt welche aufgegriffen werden', img+img_tool)

cv2.waitKey()

# %%
# Second Part  --> Creat Groups

m_type, m_orientation, m_x, m_y, m_angle = createGroup(type, orientation,x_obj,y_obj,angle,j)
print('##############################################################    Zweiter Teil')
print('Type: ', m_type)
print('Orientierung: ', m_orientation)
print('x: ',m_x)
print('y: ', m_y)
print('angle: ', m_angle)

# %%
# Third Part --> Move Parts 

alpha, m_type, error = move([1,1], [1,1],[0, 100], [0,100], [20,40], orig)
print('##############################################################    Dritter Teil')
print(alpha)
print(m_type)
print(error)


# %%


