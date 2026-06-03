"""
This sample illustrates how to get images from the blaze camera using the Python Harvester module.
"""

import os
import platform

# This is used for reshaping the image buffers.
import numpy as np
# This is used for visualization.
import cv2

# Use of Harvester to access the camera.
# For more information regarding Harvester, visit the github page:
# https://github.com/genicam/harvesters
from harvesters.core import Harvester
from datetime import datetime

# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import logo_rotation

def find_producer(name):
    """ Helper for the GenTL producers from the environment path.
    """
    paths = os.environ['GENICAM_GENTL64_PATH'].split(os.pathsep)

    if platform.system() == "Linux":
        paths.append('/opt/pylon/lib/gentlproducer/gtl/')

    for path in paths:
        path += os.path.sep + name
        if os.path.exists(path):
            return path
    return ""

try:
    h = Harvester()
    temp = True
    # Location of the Basler blaze GenTL producer.
    if platform.system() == "Windows" or platform.system() == "Linux":
        path_to_blaze_cti = find_producer("ProducerBaslerBlazePylon.cti")
    else:
        print(f"{platform.system()} is not supported")
        assert False

    os.makedirs("_2d_confidence", exist_ok=True)
    os.makedirs("_2d_intensity", exist_ok=True)
    # Add producer to Harvester.
    assert os.path.exists(path_to_blaze_cti)
    h.add_file(path_to_blaze_cti)

    # Update device list.
    h.update()

    # Print device list.
    print(h.device_info_list)

    # Connect to the first camera in the device list.
    if h.device_info_list:
        ia = h.create(0)
        print("Connected to camera: {}".format(
            ia.remote_device.node_map.DeviceSerialNumber.value))
    else:
        print("No cameras found")
        raise RuntimeError

    # In the following, we demonstrate how to get and set camera parameters.
    # For demonstration purposes and to avoid having to change the camera's state,
    # we first get a parameter value and then set it again.

    # Access OperatingMode.
    operatingMode = ia.remote_device.node_map.OperatingMode.value
    #ia.remote_device.node_map.OperatingMode.value = operatingMode
    #ia.remote_device.node_map.OperatingMode.value = 'ShortRange'
    ia.remote_device.node_map.OperatingMode.value = 'LongRange'
    print("OperatingMode: ", ia.remote_device.node_map.OperatingMode.value)

    # Access FastMode.
    fastMode = ia.remote_device.node_map.FastMode.value
    ia.remote_device.node_map.FastMode.value = fastMode
    # ia.remote_device.node_map.FastMode.value = True
    print("FastMode: ", ia.remote_device.node_map.FastMode.value)

    # Access FilterSpatial
    filterSpatial = ia.remote_device.node_map.FilterSpatial.value
    ia.remote_device.node_map.FilterSpatial.value = filterSpatial
    ia.remote_device.node_map.FilterSpatial.value = False
    # ia.remote_device.node_map.FilterSpatial.value = False
    print("FilterSpatial: ", ia.remote_device.node_map.FilterSpatial.value)

    # Access FilterTemporal
    filterTemporal = ia.remote_device.node_map.FilterTemporal.value
    ia.remote_device.node_map.FilterTemporal.value = filterTemporal
    # ia.remote_device.node_map.FilterTemporal.value = True
    print("FilterTemporal: ", ia.remote_device.node_map.FilterTemporal.value)

    # Access FilterStrength
    filterStrength = ia.remote_device.node_map.FilterStrength.value
    ia.remote_device.node_map.FilterStrength.value = filterStrength
    # FilterTemporal must be enabled before setting
    # FilterStrengh is possible.
    if ia.remote_device.node_map.FilterTemporal.value:
        ia.remote_device.node_map.FilterStrength.value = filterStrength
        # ia.remote_device.node_map.FilterStrength.value = 200
    print("FilterStrength: ", ia.remote_device.node_map.FilterStrength.value)

    # Access OutlierRemoval
    outlierRemoval = ia.remote_device.node_map.OutlierRemoval.value
    ia.remote_device.node_map.OutlierRemoval.value = outlierRemoval
    # ia.remote_device.node_map.OutlierRemoval.value = True
    print("OutlierRemoval: ", ia.remote_device.node_map.OutlierRemoval.value)

    # Access ConfidenceThreshold
    confidenceThreshold = ia.remote_device.node_map.ConfidenceThreshold.value
    ia.remote_device.node_map.ConfidenceThreshold.value = confidenceThreshold
    ia.remote_device.node_map.ConfidenceThreshold.value = 1024
    print("ConfidenceThreshold: ", ia.remote_device.node_map.ConfidenceThreshold.value)

    # Access GammaCorrection
    gammaCorrection = ia.remote_device.node_map.GammaCorrection.value
    ia.remote_device.node_map.GammaCorrection.value = gammaCorrection
    # ia.remote_device.node_map.GammaCorrection.value = True
    print("GammaCorrection: ", ia.remote_device.node_map.GammaCorrection.value)

    # 1. Get the current Exposure Time
    
    current_exposure = ia.remote_device.node_map.ExposureTime.value
    print(f"Current Exposure Time: {current_exposure} us")

    # 2. Get the allowed min and max range (Optional but safe)
    # min_exposure: 100.0 us
    # max_exposure: 1000.0 us
    min_exposure = ia.remote_device.node_map.ExposureTime.min
    max_exposure = ia.remote_device.node_map.ExposureTime.max
    print(f"Allowed Exposure Range: {min_exposure} us to {max_exposure} us")

    # 3. Set a new Exposure Time (e.g., 2000 microseconds)
    target_exposure = 600 
    
    # Ensure our target fits within the camera's current allowed bounds
    if min_exposure <= target_exposure <= max_exposure:
        ia.remote_device.node_map.ExposureTime.value = target_exposure
        print(f"Exposure Time successfully set to: {ia.remote_device.node_map.ExposureTime.value} us")
    else:
        print(f"Error: Target exposure {target_exposure} us is out of bounds!")
        print(f"min_exposure: {min_exposure} and max_exposure: {max_exposure}")
        
    # Set the working range to the values displayed for the Max. Depth [mm] and
    # Min. Depth [mm] parameters.
    # The working range depends on the current operating mode, so
    # the OperatingMode parameter must be set before adjusting DepthMax and DepthMin.
    # LongRange: 0 .. 9990mm
    # ShortRange: 0 .. 1498mm
    depthMin = ia.remote_device.node_map.DepthMin.value
    depthMax = ia.remote_device.node_map.DepthMax.value
    # ia.remote_device.node_map.DepthMin.value = depthMin
    # ia.remote_device.node_map.DepthMax.value = depthMax
    ia.remote_device.node_map.DepthMin.value = 0
    ia.remote_device.node_map.DepthMax.value = 9990
    print("Min. Depth [mm]: ", ia.remote_device.node_map.DepthMin.value)
    print("Max. Depth [mm]: ", ia.remote_device.node_map.DepthMax.value)

    # Control pixel formats for image components.
    # Range information can be sent either as a 16-bit gray value image or as
    # 3D coordinates (point cloud).
    # For this sample, we want to acquire 3D coordinates.
    # Note: To change the format of an image component, the Component Selector parameter
    # must first be set to the component
    # you want to configure.
    # To use 16-bit integer depth information, choose "Coord3D_C16" instead of "Coord3D_ABC32f".
    ia.remote_device.node_map.ComponentSelector.value = "Range"
    ia.remote_device.node_map.ComponentEnable.value = True
    ia.remote_device.node_map.PixelFormat.value = "Coord3D_ABC32f"
    #ia.remote_device.node_map.PixelFormat.value = "Coord3D_C16"

    ia.remote_device.node_map.ComponentSelector.value = "Intensity"
    ia.remote_device.node_map.ComponentEnable.value = True
    ia.remote_device.node_map.PixelFormat.value = "Mono16"

    ia.remote_device.node_map.ComponentSelector.value = "Confidence"
    ia.remote_device.node_map.ComponentEnable.value = True
    ia.remote_device.node_map.PixelFormat.value = "Confidence16"

    # Disable GenDC, this mode is currently not supported by the python genicam module
    ia.remote_device.node_map.GenDCStreamingMode.value = "Off"

    print('To exit, press ESC in one of the image windows')
    # Start image acquisition.
    ia.start()
    
    # download faceModel
    FaceModel_Path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    # load faceModel
    faceModel = YOLO(FaceModel_Path)
    handModel = YOLO(r"runs/pose/train-4/weights/best.pt")
    
    spin_angle = 0
    # --- SETUP VARIABLES ---
    text_string = "JARVIS"
    font_style = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1.3
    text_color = (212, 82, 0)  # Green text
    thickness = 4

    # Generate the initial flat text layer template
    text_layer = logo_rotation.create_text_layer(text_string, font_style, font_scale, text_color, thickness)
    # --- ADD WINDOW SIZE CONFIGURATIONS HERE ---
    # Define your target window sizes (Width, Height)
    intensity_window_width, intensity_window_height = 800, 600
    depth_window_width, depth_window_height = 400, 300

    """
    # Configure the Intensity window
    cv2.namedWindow('_2d_intensity_color', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('_2d_intensity_color', intensity_window_width, intensity_window_height)
    """
    
    # Make the display resizable window
    cv2.namedWindow('_2d_intensity_color', cv2.WINDOW_NORMAL)
    # 2. Force the window to expand into full-screen mode
    cv2.setWindowProperty('_2d_intensity_color', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    
    while True:
        with ia.fetch() as buffer:
            # Warning: The buffer is only valid in the with statement and will be destroyed
            # when you leave the scope.
            # If you want to use the buffers outside of the with scope, you have to use np.copy()
            # to make a deep copy of the image.

            # Create an alias of the image components:
            pointcloud = buffer.payload.components[0]

            intensity = buffer.payload.components[1]
            ##A visual representation that indicates the certaintly, reliability, or probability of an algorithm's predictions for every single pixel in an image
            confidence = buffer.payload.components[2]

            # Reshape the depth image into a 2D/3D array:
            # "num_components_per_pixel" depends on the pixel format selected:
            # "Coord3D_ABC32f" = 3
            # "Coord3D_C16" = 1
            #Width: 640
            #Height: 480
            _3d = pointcloud.data.reshape(pointcloud.height, pointcloud.width,
                                          int(pointcloud.num_components_per_pixel))
            # Reshape the intensity image into a 2D array:
            _2d_intensity = intensity.data.reshape(intensity.height, intensity.width)

            # Reshape the confidence map into a 2D array:
            _2d_confidence = confidence.data.reshape(confidence.height, confidence.width)

            # Show the captured images as grayscale.
            # We only show the z-component of the point cloud.
            # If you choose "Coord3D_C16" as pixel format, you have to remove [:,:,2].
            # OpenCV can't show float values. We convert it for visualization to uint8.
            # dtype: uint8
            _3d_scaled = _3d * 255.0 / ia.remote_device.node_map.DepthMax.value
            #cv2.imshow('depth', _3d_scaled[:, :, 2].astype(np.uint8))
            
            # dtype: uint16
            #cv2.imshow('_2d_intensity', _2d_intensity)

            # dtype: uint16
            cv2.imshow('_2d_confidence', _2d_confidence)
            
            rgb_like_intensityImg = cv2.cvtColor(_2d_intensity, cv2.COLOR_GRAY2RGB)
            #convert uint16 to uint8
            rgb_like_intensityImg = (rgb_like_intensityImg/256).astype(np.uint8)
            
            rgb_like_confidenceImg = cv2.cvtColor(_2d_confidence, cv2.COLOR_GRAY2RGB)
            rgb_like_confidenceImg = (rgb_like_confidenceImg/256).astype(np.uint8)
            
            #cv2.imshow('_2d_intensity_color', rgb_like_intensityImg)
            
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
            outputs = faceModel(rgb_like_intensityImg, conf=0.5)
            results_head = Detections.from_ultralytics(outputs[0])
            
            results_hand = handModel(rgb_like_confidenceImg, iou=0.45, conf=0.25)

            # Progress spinning angle step (Increase to spin faster, decrease to slow down)
            spin_angle = (spin_angle + 10) % 360
            
            # Process current horizontal tilt layer frame (fov=250 provides clean depth perspective)
            rotated_text = logo_rotation.get_horizontally_rotated_layer(text_layer, spin_angle, fov=250)
            th, tw = rotated_text.shape[:2]

            
            #call all detected boxes
            for i in range(len(results_head)):
                
                result = results_head[i]

                boxes = result.xyxy
                scores = result.confidence
                classes = result.class_id
                
                print("Result %s" %len(results_head))
                        
                for box, score, cls in zip(boxes, scores, classes):
                    
                    print(f"Object: {cls} | Bounding Box: {box.tolist()} | Confidence Score: {score:.2f}")

                    cords = box.tolist()
                    x1, y1, x2, y2 = [int(x) for x in cords]
                    box = box.astype(int)
                    #cv2.rectangle(rgb_like_intensityImg, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            
                    #Text:Confidence level
                    text = f"{score:.2f}"
                    coordinates = (int(box[0]), int(box[1])-20)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255 , 0, 0) #Blue
                    thickness = 2
                    
                    #cv2.putText(rgb_like_intensityImg,  text, coordinates, font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    #Text:Coordinate Value
                    x_px_center = int((box[0]+box[2])/2)
                    y_px_center = int((box[1]+box[3])/2)
                    text = f"({x_px_center}, {y_px_center})"
                    
                    text = f"Welcome to Jarvis!"
                    coordinates = (int(box[0] - 100), int(box[1] - 20))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255 , 0, 0) #Blue
                    thickness = 2
                    
                    #cv2.putText(rgb_like_intensityImg,  text, coordinates, font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    #Dot on an input image

                    dot_center = (int(x_px_center), int(y_px_center))
                    dot_radius = 5
                    dot_color = (255,0,0) #Blue
                    dot_thinkness = -1
                    
                    #cv2.circle(rgb_like_intensityImg, dot_center, dot_radius, dot_color, dot_thinkness)
                    
                    #_3D image xy coordinate: center of the image. 
                    x_coord = _3d[y_px_center, x_px_center, 0]
                    y_coord = _3d[y_px_center, x_px_center, 1]
                    z_depth = _3d[y_px_center, x_px_center, 2]
                    
                    # Print out the distance in millimeters (mm)
                    print(f"Coordinates at ({x_px_center}, {y_px_center}) -> X: {x_coord:.2f}mm, Y: {y_coord:.2f}mm, Depth (Z): {z_depth:.2f}mm")

                    x_off = x_px_center 
                    y_off = y_px_center - 80
                    
                    roi = rgb_like_intensityImg[y_off - int(th/2):y_off + int(th/2) if y_off+int(th/2) <= intensity.height else intensity.height , x_off - int(tw/2):x_off+int(tw/2) if x_off+int(tw/2) <= intensity.width else intensity.width]
                    print(f"y_off: {y_off} x_off: {x_off} th: {th} tw: {tw}")
                    
                    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                    #2 Extract the actual height and width
                    roi_h, roi_w = roi.shape[0], roi.shape[1]
                    
                    #3. Crop your roated text to Match the ROI dimensions exactly
                    cropped_rotated_text = rotated_text[0:roi_h, 0:roi_w]
                    
                    gray = cv2.cvtColor(cropped_rotated_text, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    
                    # Segment out foreground and background, then combine them seamlessly
                    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    fg = cv2.bitwise_and(cropped_rotated_text, cropped_rotated_text, mask=mask)
                    
                    rgb_like_intensityImg[y_off - int(th/2):y_off + int(th/2) , x_off - int(tw/2):x_off+int(tw/2)] = cv2.add(bg, fg)
                    #rgb_like_intensityImg[y_off - int(th/2):y_off + int(th/2) , x_off - int(tw/2):x_off+int(tw/2)] = fg
                    
            print(f"The number of output: {len(outputs)}")  
            """
            for result in results_hand :

                boxes = result.boxes
                keypoints = result.keypoints.xy.cpu().numpy()
                
                for box, keypoint in zip(boxes, keypoints):
                    
                    boxX1, boxY1, boxX2, boxY2 = box.xyxy[0].tolist()
                    #middle_finger_mcp = keypoint[9]
                    Index_Finger_Tip = keypoint[8]
                    tipX, tipY = Index_Finger_Tip.tolist()

                    print(f"box x1: {boxX1} middle_finger_mcp: {Index_Finger_Tip}")
                    
                    #Index_Finger_Tip dot circle on rgb intensity image
                    dot_center = (int(tipX), int(tipY))
                    dot_radius = 5
                    dot_color = (0,0,255) #Blue
                    dot_thinkness = -1

                    #cv2.circle(rgb_like_intensityImg, dot_center, dot_radius, dot_color, dot_thinkness)
                """
            cv2.imshow('_2d_intensity_color', rgb_like_intensityImg)
            cv2.imshow('depth', _3d_scaled[:, :, 2].astype(np.uint8))

            #cv2.imwrite(fr"_2d_intensity/{now}.png", _2d_intensity)
            #cv2.imwrite(fr"_2d_confidence/{now}.png", _2d_confidence)

        # Break the endless loop by pressing ESC.
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # Stop image acquisition.
    ia.stop()

    # Disconnect from camera.
    ia.destroy()

finally:
    # Remove the CTI file and reset Harvester.
    h.reset()
