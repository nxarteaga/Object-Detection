import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

print(torch.backends.mps.is_available())
model  = YOLO("yolov8m.pt")
#print(cv2.getBuildInformation()) #Prints OpenCv information for Debug purposes
#print("OpenCV version:", cv2.__version__)
video_dir = '/Users/arteaga/anaconda3/envs/cv/'
#print(os.listdir(video_dir)) #OS function to list directory files

# Use the absolute path to the video file
video_path = '/Users/arteaga/anaconda3/envs/cv/items.mov'
if not os.path.exists(video_path):
    print(f"Error: The file {video_path} does not exist.")
else:
    cap = cv2.VideoCapture(video_path)
    print(f"VideoCapture object: {cap}")

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        #While loop can go here to open video frame by frame. 
        # Read a frame from the video
        while True:  
            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame, device="mps")
            result = results[0]
            boundingBoxes = np.array(result.boxes.xyxy.cpu(),dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            for cls, bbox in zip(classes, boundingBoxes):
                (x, y, x2, y2) = bbox

                 
                cv2.rectangle(frame, (x, y), (x2, y2),(0, 0, 225), 2)
                cv2.putText(frame, str(cls),(x, y -5), cv2.FONT_HERSHEY_PLAIN, 1,(225),2)
            #print(boundingBoxes)
            #print(results)
            # Display the frame in a window named "Img"
            cv2.imshow("Img", frame)
            # Wait for a key press to close the window
            key = cv2.waitKey(1)
            # Destroy the window
            #cv2.destroyAllWindows()
           # else:
                #break
                #print("Error: Could not read frame from video.")

            if key == 27:
                break


    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()