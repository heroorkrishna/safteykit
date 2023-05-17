import torch
import numpy as np
import cv2
import time
from time import time
from ultralytics import YOLO
from supervision.draw.color import ColorPalette,Color
from supervision.tools.detections import Detections, BoxAnnotator
import pyglet #for sound alert

import mysql.connector

# database connection
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="meonly",
  database="helmetdetectiondb"
)

cursor = db.cursor() 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

model = YOLO('C:/Users/admin/Downloads/AdityaPOC/new_saftey_helmet_detection/Safety_kit_det/yolos_250epoch.pt')
model.fuse()
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

#Not permitted area coordinates where without helmet person not allowed
area_1 = [(587,287),(617,286),(422,722),(375,682)]  # camera right coordinates
area_2 = [(810,231),(881,779),(1019,779),(854,256)] # camera left coordinates

def plot_bboxes(results,frame,counter):
    xyxys = []
    confidences = []
    class_ids = []

    boundary_box_camera_right = [(810,231),(881,779),(1019,779),(854,256)]   #camera right coordinates
    boundary_box_camera_left = [(587,287),(617,286),(422,722),(375,682)]     #camera left coordinates

    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id == 5:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            cv2.putText(frame, "Warning!..Please Wear Safety Helmet", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if counter % 100 == 0 or counter ==20:
                sound_alert(frame)

        elif class_id == 4:
            confidences.append(result.boxes.conf.cpu().numpy())
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            cv2.putText(frame, "Warning!..Please Wear Hand Gloves", (50,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if counter % 150 == 0 or counter ==20:
                sound_alert(frame)
        elif class_id == 6:
            confidences.append(result.boxes.conf.cpu().numpy())
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            cv2.putText(frame, "Warning!..Person Down", (50,360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
        elif class_id == 7:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            if is_crossing_boundary_box(xyxys, boundary_box_camera_right,boundary_box_camera_left,frame):
                print(frame)

    detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )
    
    # Format custom labels
    labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections]
    
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    
    return frame

## code for triggering alert when person touches/crosses the alert coordinates

def is_crossing_boundary_box(xyxy, boundary_box_camera_right, boundary_box_camera_left, frame):
    for box in xyxy:
        x1, y1, x2, y2 = box[0]
        if x1 >= boundary_box_camera_right[0][0] and x2 <= boundary_box_camera_right[2][0] and y1 >= boundary_box_camera_right[0][1] and y2 <= boundary_box_camera_right[2][1]:
            print("Warning! Person crossed the boundary box (camera right)")
            cv2.putText(frame, "Warning! Person crossed the boundary box (camera right)", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return True
        elif x1 >= boundary_box_camera_left[0][0] and x2 <= boundary_box_camera_left[2][0] and y1 >= boundary_box_camera_left[0][1] and y2 <= boundary_box_camera_left[2][1]:
            print("Warning! Person crossed the boundary box (camera left)")
            cv2.putText(frame, "Warning! Person crossed the boundary box (camera left)", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return True
    return False

def sound_alert(frame):
        saved_nohelmetimages_counter=1
        music_voice = pyglet.resource.media('beep_alert_ok.mp3')
        music_voice.play()
        cv2.imwrite('Saved_NoHelmet/'+str(saved_nohelmetimages_counter)+'.jpg',frame)

def stream_vid(capture_index):    
    counter=0
    cap = cv2.VideoCapture(capture_index)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        start_time = time()
        ret, frame = cap.read()
        assert ret

        frame=cv2.resize(frame,(1500,900))

        results = model(frame)
        counter=0 if counter == 2000 else counter+1
  
        frame =  plot_bboxes(results, frame,counter)
        

        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)


        cv2.putText(frame, f'FPS: {int(fps)}', (10,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        
        cv2.polylines(frame,[np.array(area_1,np.int32)],True,(255,0,0),3)
        cv2.polylines(frame,[np.array(area_2,np.int32)],True,(255,0,0),3)
        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture_index='C:/Users/admin/Downloads/AdityaPOC/new_saftey_helmet_detection/Safety_kit_det/video_tmp17.mp4'
#capture_index='videos/video_tmp/poc/video_tmp12.mp4'
#capture_index='videos/video_tmp/poc/video_tmp15.mp4'
#ip_link='rtsp://devanshu:admin123@192.168.10.49:554/401'
stream_vid(capture_index)
