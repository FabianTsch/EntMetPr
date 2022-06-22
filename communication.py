"""
Created on Fr May  20 15:43:00 2022

@author: Stefan Kaufmann
Abschlussprojekt

Communication: Filters the array of object checks for collisions 
"""

# %%
import cv2
import numpy as np

plot = False
debug = True



def createObj(obj_type, orientation,x,y,angle, orig, tool=False,move=False, workspace=100):
    """Creats a obj, screw nut, lying/standing 
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
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
        img:           binary image with 1 = obj ,  0 = no obj,   image is a np.array with dim of (orig)         
                
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
        if obj_type == 1 and orientation == 2:  # tool for screw 
            width = 42+s
            height = 25+s
           
                        
        else:   # tool for nuts
            width = 42+s
            height = 20+s     
   
    elif obj_type == 1 and orientation == 2:   # screws
        width = 20
        height = 50
    else:                                  # nuts
        width = 20
        height = 20
    
    if move:       
        width = 42+s
        height = 30+s

    obj = np.ones((height,width))
    if tool and not move:
        obj[:,2*s:width-2*s] = 0

            
    
    # Rotate the img
    w = int(np.sqrt(np.square(height)+np.square(width)))                        # new size of the img

    M = np.float32([[1,0,int((w-width)/2)],[0,1,int((w-height)/2)]])            # translation to the new center
    obj = cv2.warpAffine(obj, M, (w, w))

    rows,cols = obj.shape
    M = cv2.getRotationMatrix2D((w/2,w/2), angle, 1)
    obj = cv2.warpAffine(obj, M, (w, w))
    

    # Draw the obj in the work space 
    rows,cols = obj.shape   
    
    for i in range(0,rows):
        for j in range(0,cols):
            if obj[i,j] == 1 or img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] == 1:        # if obj or img at the location of the obj is 1 --> img at the location is 1
                img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] = 1                 

    if plot:
         return img 
          
    return img[workspace:orig[1]+workspace, workspace:orig[0]+workspace]

def createImg(obj_type, orientation,x,y,angle, orig,tool=False,move = False, workspace=100):
    """Creats a obj, screw nut, lying/standing, can handle scalars and arrays
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        tool:          if we want to grab with tool   # true = tool, false = no tool    default False 
        move:          if we want to move the obj with a tool, # true = move
        workspace:     work space is the overlay of the table, we need it for the implementation,  default 100 mm

        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no ob, image is a np.array with dim of (orig)   with all given obj in it         
                
    """
    if plot:
        img = np.zeros((orig[1]+2*workspace, orig[0]+2*workspace))
    else:
        img = np.zeros((orig[1], orig[0]))

    if np.isscalar(obj_type):
        img = createObj(obj_type,orientation,x,y,angle, orig, tool,move, workspace)
    else:
        for i in range(0,np.size(obj_type)):
            img += createObj(obj_type[i],orientation[i],x[i],y[i],angle[i], orig, tool,move, workspace)           
    
    return img

def pick(obj_type, orientation,x,y,angle,img,move=False, workspace=100):
    """ Checks if the obj can be picked
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        img:           Binary image with all remaining Objects 1 = object  
        move:          if we want to move the obj with a tool, # true = move                          

        Returns
        --------
        posible:       True if the obj can be picked   --> scalar        
                
    """
    # creats a tool img
    y_img,x_img = img.shape
    
    if plot:
        y_img -= workspace*2
        x_img -= workspace*2
    img_tool = createObj(obj_type, orientation,x,y,angle, (x_img,y_img), True, move)
    if plot:
        cv2.imshow('img+tool', img+img_tool)
        cv2.waitKey()
    img_bitwise_and = cv2.bitwise_and(img, img_tool)
    if img_bitwise_and.any():
        possible = False
    else:
        possible = True

    if plot:
        img_bitwise_and = img_bitwise_and[(workspace+1):y_img+workspace, (workspace+1):x_img+workspace]  # shrinks the image to the area of ​​interest
        if img_bitwise_and.any():
             possible = False
        else:
            possible = True
        

    return possible

def checkPick(obj_type, orientation,x,y,angle,orig, img = 0):
    """ Checks if the obj can be picked works with scalars and arrays
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        img:           Binary image with all remaining Objects 1 = object, by default creats his own binary
                          

        Returns
        --------
        j:             Array of indixes of the obj which can be picked in the correct order  
        img:           binary image with 1 = obj ,  0 = no ob, image is a np.array with dim of (orig)   with all obj who can't be picked       
                
    """
    # Create reconstructed image
    if img == 0:
        img = createImg(obj_type, orientation,x,y,angle, orig)       
       
    if plot:
        cv2.imshow('Das Orginal', img) 
    # Helpful variables
    possible = []
    j = []                  # Array of indixes of the obj which can be picked in the correct order

    if np.isscalar(obj_type): 
        possible = pick(obj_type, orientation,x,y,angle,img)
        return possible
    
    
    else:
        for count in range(0,np.size(obj_type)):  # do the check as often as there are elements in obj  
            # if plot:
            #     print ('#################  Durlauf Nr.: ', count)            

            for i in range(0,np.size(obj_type)):  # first loop over all objects  
                if count%2 == 0:
                    i = np.size(obj_type)-i -1                
      
                if not i in j and pick(obj_type[i], orientation[i],x[i],y[i],angle[i],img):   # Controls if any i ist part of j and if obj[i] is pickable            
                                   
                    j = np.append(j,i)   
                    img -=createImg(obj_type[i], orientation[i],x[i],y[i],angle[i],orig) # subtracts the object from the image       
                    
            
    
    return j, img

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
    
    #print(near)
    return near

def createSubGroup(obj_type, orientation,x,y,angle, area=0):
    """Creats a Group of obj with are neare
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad    
        area:          area of the given objects in mm²   
       
        Returns
        --------
        g_x:           x Position of the Group in mm from the origin     --> Array
        g_y:           y Position of the Group in mm from the origin     --> Array
        g_angle:       Orientation of the main axis in grad              --> Array  
        g_area:        area of the group in mm²                          --> Array     
                
    """
    g_x = []
    g_y = []
    g_angle = [] 
    g_area = []
    g_j = np.array([])            # inidizes of whitch obj that are not seperated
    g_type = np.zeros(np.size(x))     # helpful variable to declare if an obj is part of a Group during the loop
    
    # Error handling if we get an scalar instead of an list    
    if  np.size(x) == 1:
        return x, y, angle, area
         
    if np.isscalar(area):                         # For the first loop
        surface = np.ones(np.size(obj_type))   

        
        for i in range(0,np.size(obj_type)):
            if obj_type[i] == 1 and orientation[i] == 2:
                surface[i] = 2
            elif obj_type[i] == 2 and orientation[i] == 1:
                surface[i] == 0.5        
    else:
        surface = area
    

    # creat group 
       
    for i in range(0,np.size(x)):
        #print('************************* i:',i)
        
        for j in range(0,np.size(x)):
            
            if nearObj(x[j], y[j], x[i],y[i]) and not int(g_type[i]) == 3 and not int(g_type[j]) == 3 and not i == j:
                print('Obj erstellt ', j)
                
                # common parameters based on their areas 
                g_x = np.append(g_x,np.abs((x[j]*surface[j]+x[i]*surface[i])/(surface[j]+surface[i])))
                g_y = np.append(g_y,np.abs((y[j]*surface[j]+y[i]*surface[i])/(surface[j]+surface[i])))
                g_angle = np.append(g_angle,np.abs((angle[j]*surface[j]+angle[i]*surface[i])/(surface[j]+surface[i])))
                g_area = np.append(g_area,surface[j]+surface[i] )            
                g_type[i] = 3 
                g_type[j] = 3                 

            elif  not i  in g_j and not j in g_j:    
                g_j = np.append(g_j,i)

               
    # attach  
    if not np.size(g_j) == 0:              
        for i in g_j:
            i = int(i)
            if not int(g_type[i]) == 3:
                #print('attach',i)            
                g_x = np.append(g_x, x[i] )  
                g_y = np.append(g_y, y[i] ) 
                g_angle = np.append(g_angle, angle[i]) 
                g_area = np.append(g_area, surface[i])

    #print('type ', g_type)
    #print('g_j:',g_j)
    #print('Ende creatGroup')
    return g_x, g_y, g_angle, g_area

def createGroup(obj_type, orientation,x,y,angle,j):
    """Creats a Group of obj with are neare
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
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
    

    #print('Creat Group')
    # creat a list with obj that can not be picked
    for i in range(0,np.size(obj_type)):

        if not i in j:
            m_type = np.append(m_type, obj_type[i])
            m_orientation = np.append(m_orientation, orientation[i])
            m_x = np.append(m_x, x[i])
            m_y = np.append(m_y, y[i])
            m_angle = np.append(m_angle, angle[i])   
                  
    

    # first loop
    m_x, m_y, m_angle, m_area = createSubGroup(m_type,m_orientation,m_x,m_y,m_angle)   
    
    # other loops
    if not np.size(m_x)==1:
        for i in range(0,np.size(obj_type)):
            # if plot:
            #     print('####################### Runde :', i)
            m_x, m_y, m_angle, m_area = createSubGroup(m_type,m_orientation,m_x,m_y,m_angle,m_area)
        
     
    
    m_orientation = np.ones(np.size(m_x)) *2    # Orientation chanced to lying
    
    return m_type, m_orientation, m_x, m_y, m_angle
 
def move(obj_type, orientation,x, y, angle, orig, img):
    """Creats the attack angle
        Params
         --------
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad   
        
       
        Returns
        --------
        
        m_angle:       Orientation of the attack angle of the group  
        m_type:        set obj_type seperate    
        error:         if there is error, 1 = no rechable Positionen found          
                
    """  
   

    # check if their are more than one group
    if np.size(x) == 1:
        
        alpha = angle
        m_x = x   
        m_y = y
        

        while not pick(1,2,x,y,alpha[0],img,True):           
            m_x -= np.cos(np.deg2rad(alpha)) * 1
            m_y -= np.sin(np.deg2rad(alpha)) * 1
        
        m_x -= np.cos(np.deg2rad(alpha)) * 15
        m_y -= np.sin(np.deg2rad(alpha)) * 15
           
        return 3, 1, m_x,m_y,360-alpha, 0


    # important variables    
    alpha = np.zeros(np.size(x))
    error = np.zeros(np.size(x))
    m_x = np.zeros(np.size(x))
    m_y = np.zeros(np.size(x))    
   
    # find nearts group    
    for count in range (0,np.size(x)):        
        r = 0
        z = 0
        threshold = np.inf
        # first step, check for the nearest group
        for i in range(0,np.size(x)):
            if not count == i:    
                r = ((x[count]-x[i])**2+(y[count]-y[i])**2)**0.5             
                if r <= threshold:
                    threshold = r + 0
                    j = i   # nearest group
        
        # second step search for the ideal attack vector
        # creat the vektor between the two Groups
        m_alpha = np.arctan2((y[j]-y[count]),(x[j]-x[count]))  


        # check if this position is reachable      
        x_m = r*np.cos(m_alpha)/2 + x[count]
        y_m = r*np.sin(m_alpha)/2 + y[count] 
        m_alpha = m_alpha*180/np.pi
        while not pick(1,2,x_m,y_m,m_alpha,img,True):  # if alpha is not poissible --> rotate alpha 
            m_alpha += 5  
            z += 5
            #print('+z')
            if z > 450:
                x_m += np.cos(np.deg2rad(m_alpha)) * 10
                y_m += np.sin(np.deg2rad(m_alpha)) * 10
                i = 0
                
                print('Error --> keine erreichbare Position gefunden, Radius wird vergrößert')
                error[count] = 1
                #break 
        
        # find the distance to the obj make steps of 1 mm
        while pick(1,2,x_m,y_m,m_alpha,img,True):
            x_m -= np.cos(np.deg2rad(m_alpha)) * 1
            y_m -= np.sin(np.deg2rad(m_alpha)) * 1

        x_m += np.cos(np.deg2rad(m_alpha)) * 15
        y_m += np.sin(np.deg2rad(m_alpha)) * 15

        
        if m_alpha > 360:
            m_alpha -= 360 * (m_alpha//360)
           

        if m_alpha <0:
            m_alpha = abs(m_alpha)+180
        elif m_alpha<180:
            m_alpha = 180-m_alpha
       

        alpha[count] = m_alpha
        m_x[count] = int(x_m)
        m_y[count] = int(y_m)
 


    m_type = np.ones(np.size(x)) *3           # set obj_type seperate

    return m_type, orientation,m_x,m_y, alpha, error

def execute(obj_type, orientation,x,y,angle, orig=[750,500]):
    """Execute file for creating list of objs
        Params
         --------
        obj_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image   
                
        Returns
        --------
        r_type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        r_orientation:   standing = 1, lying = 2,  perhabs more info    
        r_x:             Position from origin in mm - x-Position
        r_y:             Position from origin in mm - y-Position
        r_angle:         Orientation of the main axis in grad  
        r_orig:          Original dimensions [x,y] of the image   
                      
                
    """
    # Return Values
    
    r_type = []
    r_orientation = []
    r_x = []
    r_y = []
    r_angle = []

    for i in range(0,len(angle)):
        if angle[i] <0:
            angle[i] = angle[i] + 360            
    

    # First Part -  check pick
    j, img = checkPick(obj_type, orientation,x,y,angle,orig)

    if debug:
        offset = 0
        if plot:
            offset = 100

        img_text = createImg(obj_type, orientation,x,y,angle, [750,500])
        img_text = img_text.astype(np.float32)*255    
        img_text = cv2.cvtColor(img_text, cv2.COLOR_GRAY2RGB)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in j:
            i = int(i)
            cv2.putText(img_text, str(i),(x[i]+offset,y[i]+offset), font,
                            1, (0, 0, 255), 2)
        cv2.imshow('alle Objekte', img_text)


    # Second Part -  Creat Groups
    m_type, m_orientation, m_x, m_y, m_angle = createGroup(obj_type, orientation,x,y,angle,j)
        
    # Third Part Find attack angle
    a_type, a_orientation, a_x, a_y, a_angle, error = move(m_type, m_orientation,m_x.copy(), m_y.copy(), m_angle.copy(), orig, img)
   
    if debug:
        offset = 0
        if plot:
            offset = 100   

        img_text =  img        
        img_text = img_text.astype(np.float32)*255    
        img_text = cv2.cvtColor(img_text, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(0,np.size(m_x)):
            i = int(i)   
            img_text = cv2.line(img_text, (int(a_x[i]+offset),int(a_y[i]+offset)), (int(m_x[i]+offset), int(m_y[i]+offset)), (0,255, 0), thickness=2)     
            cv2.putText(img_text, str(int(a_angle[i])),(int(a_x[i]+offset),int(a_y[i]+offset)), font, 0.5, (0, 255, 0), 2)                      
                
            cv2.putText(img_text, str(i),(int(m_x[i]+offset),int(m_y[i]+offset)), font,1, (0, 0, 255), 2)
                
        cv2.imshow('Gruppen', img_text)
        cv2.waitKey()
    # Fouth Part - Creat list
    for i in range(0,np.size(j)):
        r_type = np.append(r_type, obj_type[int(j[i])])        
        r_orientation = np.append(r_orientation, orientation[int(j[i])])       
        r_x = np.append(r_x, x[int(j[i])])       
        r_y = np.append(r_y, y[int(j[i])])        
        r_angle = np.append(r_angle, angle[int(j[i])])
      
    r_type = np.append(r_type,a_type)       
    r_orientation = np.append(r_orientation, a_orientation)
    r_x = np.append(r_x,a_x)
    r_y = np.append(r_y,a_y)
    r_angle = np.append(r_angle, a_angle)
    
    r_y = 500 - r_y  # Spiegeln an der y-Achse

    return r_type, r_orientation, r_x, r_y, r_angle, error

     




# %%

# ****************************************************************************************************************************************************************
def test():
    # ****************************************************************************************************************************************************************
    #For Testing
    # image size
    x = 750          
    y = 500

    obj_type = [1,1,1,1]
    orientation = [1,2, 1, 2]
    x_obj = [0,5, 200, 200]
    y_obj = [0,5, 200, 200]
    angle = [0,0, 0, 20]
    orig = [x,y]

    img = createImg(obj_type, orientation,x_obj,y_obj,angle, orig)
    img_tool =  createImg(obj_type, orientation,x_obj,y_obj,angle, orig, True )



    # %%
    #First Part --> seperating all objetcs who can be picked
    print('##############################################################    Erster Teil')
    j, img_check = checkPick(obj_type, orientation,x_obj,y_obj,angle,orig)
    print('Folgende Indizies sind aufgreifbar: ', j)
    cv2.imshow('alle Objekt', img+img_tool)
    img_tool =  np.zeros((orig[1], orig[0]))
    img = np.zeros((orig[1], orig[0]))
    for i in j:
        i = int(i)
        img += createImg(obj_type[i], orientation[i],x_obj[i],y_obj[i],angle[i], orig )
        img_tool += createImg(obj_type[i], orientation[i],x_obj[i],y_obj[i],angle[i], orig, True )    


   

    # %%
    # Second Part  --> Creat Groups

    m_type, m_orientation, m_x, m_y, m_angle = createGroup(obj_type, orientation,x_obj,y_obj,angle,j)
    print('##############################################################    Zweiter Teil')
    print('Type: ', m_type)
    print('Orientierung: ', m_orientation)
    print('x: ',m_x)
    print('y: ', m_y)
    print('angle: ', m_angle)

    # %%
    # Third Part --> Move Parts 

    m_type, orientation,x,y, alpha, error = move([1,1], [1,1],[0, 100], [0,100], [20,40], orig, img)
    print('##############################################################    Dritter Teil')
    print(alpha)
    print(m_type)
    print(error)

def test2():
    # %%
    y_max = 500
    x_max = 750
    
    obj_class = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2]
    orientation = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    x = [345, 256, 144,  71, 449, 163, 347,  84, 263, 406, 251, 430, 138, 324, 339, 498, 495]
    y = [445, 423, 415, 410, 370, 356, 337, 333, 246, 235, 233, 228, 215, 184, 172, 140, 121]
    angle = [350.4, 348.4, 354.3, 323.8, 224.,  335.2, 182.2 ,333.4,523.6 ,510.9 ,530.4 ,515.1, 331.2, 533.7, 182.8, 190.3 ,188.2]
    
    j, img = checkPick(obj_class, orientation,x,y,angle,[750,500])

    # Show the Numbers of the image
    img_text = createImg(obj_class, orientation,x,y,angle, [750,500])
    img_text = img_text.astype(np.float32)*255    
    img_text = cv2.cvtColor(img_text, cv2.COLOR_GRAY2RGB)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in j:
        i = int(i)
        cv2.putText(img_text, str(i),(x[i],y[i]), font,
                        1, (0, 0, 255), 2)
    cv2.imshow('alle Objekte', img_text)
    #cv2.waitKey()
    
    m_type, m_orientation, m_x, m_y, m_angle = createGroup(obj_class, orientation,x,y,angle,j)    
    a_type, a_orientation, a_x, a_y, a_angle, error = move(m_type, m_orientation,m_x, m_y, m_angle, [750,500], img)
    print('Angle: ', a_angle)
    print(' x: ', a_x,' y:', a_y )


   
    
    img_text =  img
    # Flip the img upside down for better interpretation
    #img_text = cv2.flip(img_text, 0)   
    img_text = img_text.astype(np.float32)*255    
    img_text = cv2.cvtColor(img_text, cv2.COLOR_GRAY2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0,np.size(m_x)):
        i = int(i)   
        img_text = cv2.line(img_text, (int(a_x[i]),int(a_y[i])), (int(m_x[i]), int(m_y[i])), (0,255, 0), thickness=1)     
        cv2.putText(img_text, str(int(a_angle[i])),(int(a_x[i]),int(a_y[i])), font, 0.5, (0, 255, 0), 2)                      
              
        cv2.putText(img_text, str(i),(int(m_x[i]),int(m_y[i])), font,1, (0, 0, 255), 2)
    
    #cv2.imshow('Gruppen', img_text)
    #img_text = cv2.flip(img_text, 0)   
    
    cv2.imshow('Gruppen- gespiegelt', img_text)
    cv2.waitKey()


    

# %%
#test2()