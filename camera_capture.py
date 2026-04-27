from ids_peak import ids_peak as peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl
import numpy as np
import cv2


def capture_image():
    peak.Library.Initialize() #Initialize the camera


    device_manager = peak.DeviceManager.Instance()# Create a device manager object
    device_manager.Update()#Refresh to confirm the device

    if device_manager.Devices().empty():
        raise RuntimeError("No IDS camera found")
   
    #Find the camera in the whole system, make the camera in the control mode
    device = device_manager.Devices()[0].OpenDevice(peak.DeviceAccessType_Control)
    
    #Open the data stream for the camera to make it upload the image data to the computer
    datastream = device.DataStreams()[0].OpenDataStream()

    #import to control the parameter for the camera
    nodemap = device.RemoteDevice().NodeMaps()[0]
    # ===== Camera basic settings =====
    nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
    nodemap.FindNode("AcquisitionMode").SetCurrentEntry("SingleFrame")

    # Set manual parameters
    nodemap.FindNode("ExposureTime").SetValue(1000.131)
    nodemap.FindNode("Gain").SetValue(15.10)

    nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
    nodemap.FindNode("UserSetLoad").Execute()
    nodemap.FindNode("UserSetLoad").WaitUntilDone()

    nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
    nodemap.FindNode("AcquisitionMode").SetCurrentEntry("SingleFrame")
    
    
    #Read how large the image is, make the computer to leave the space
    payload_size = nodemap.FindNode("PayloadSize").Value()

    # Allocate required buffers
    buffer_count = datastream.NumBuffersAnnouncedMinRequired()
    for i in range(buffer_count):
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)

    # Lock transport layer parameters before acquisition
    nodemap.FindNode("TLParamsLocked").SetValue(1)

    # Start host side acquisition first
    datastream.StartAcquisition()
    
    nodemap.FindNode("AcquisitionStart").Execute()
    nodemap.FindNode("AcquisitionStart").WaitUntilDone()
    #Wait for 5 seconds for the image capture and save the data into the buffer
    buffer = datastream.WaitForFinishedBuffer(5000)
   
    #Convert buffer into image
    image = ids_peak_ipl_extension.BufferToImage(buffer)

    #Convert image into numpy mode for further image process
    image_np = image.get_numpy_1D()

    row_size = len(image_np) // image.Height()
    image_np = image_np.reshape((image.Height(), row_size))
    image_np = image_np[:, :image.Width()]
    datastream.QueueBuffer(buffer)

    # Save the picture for the current attempt
    cv2.imwrite("current_capture.jpg", image_np)

   #Stop capture picture
    nodemap.FindNode("AcquisitionStop").Execute()
    datastream.StopAcquisition()
    nodemap.FindNode("TLParamsLocked").SetValue(0)

    peak.Library.Close()

    return "current_capture.jpg"
    return image_np #return the current picture into main file
