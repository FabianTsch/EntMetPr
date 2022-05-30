# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:37:19 2022
edit on Th. May 19 2022

@author: c8501053
@edit by: csb1729
"""
import socket
import time
import struct

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#create TCP client
# Connect the socket to the port where the server is listening
server_address = ('138.232.72.123', 6969)
sock.connect(server_address)

#---------------------------------------------------


#---------------------------------------------------


def tcp_communication(type, orientation,x,y,phi):

    """Sends the Params to the Stäubli and waits for response
        can handle arrays and scalars
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        phi:           Orientation of the main axis in grad                         

        Returns
        --------
        Error:         0 = no error, 1 = error
                
     """

    for i in range(len(x)):
        # put data in array for sending
        data=struct.pack('ddddd',type[i],orientation[i],x[i] ,y[i], phi[i])
        # send Data
        sock.sendall(data)
        
        while not (sock.recv(1)):
            print('waiting for ready signal...')
            time.sleep(0.5)
        print('Robot is ready')
        
    print('Done!')

    return 0
