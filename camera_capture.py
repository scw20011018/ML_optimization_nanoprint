import ids_peak
import ids_peak_ipl
import ids_peak_ipl_extension
import numpy as np
import cv2


def capture_image():

    ids_peak.Library.Initialize() #Initialize the camera

    device_manager = ids_peak.DeviceManager.Instance()# Create a device manager object
    device_manager.Update()#Refresh to confirm the device

    if device_manager.Devices().empty():
        raise RuntimeError("No IDS camera found")
   
    #Find the camera in the whole system, make the camera in the control mode
    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    
    #Open the data stream for the camera to make it upload the image data to the computer
    datastream = device.DataStreams()[0].OpenDataStream()

    #import to control the parameter for the camera
    nodemap = device.RemoteDevice().NodeMaps()[0]

    #Read how large the image is, make the computer to leave the space
    payload_size = nodemap.FindNode("PayloadSize").Value()

    #Allocate the buffer, record the data here
    buffer = datastream.AllocAndAnnounceBuffer(payload_size)
    datastream.QueueBuffer(buffer)

    #Start to capture picture, make sure the camera is ready to take photo
    nodemap.FindNode("AcquisitionStart").Execute()
    nodemap.FindNode("AcquisitionStart").WaitUntilDone()

    #Wait for 5 seconds for the image capture and save the data into the buffer
    buffer = datastream.WaitForFinishedBuffer(5000)

    #Convert buffer into image
    image = ids_peak_ipl_extension.BufferToImage(buffer)

    #Convert image into numpy mode for further image process
    image_np = image.get_numpy_1D()#operate in line
    image_np = image_np.reshape((image.Height(), image.Width()))#operate in 2*2 matrix

    # Save the picture for the current attempt
    cv2.imwrite("current_capture.jpg", image_np)

    #Stop capture picture
    nodemap.FindNode("AcquisitionStop").Execute()

    ids_peak.Library.Close()

    return image_np #return the current picture into main file