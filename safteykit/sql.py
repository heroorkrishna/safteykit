import torch
import numpy as np
import cv2


import time
from time import time
from ultralytics import YOLO
import mysql.connector
import datetime
import base64

from supervision.draw.color import ColorPalette,Color
from supervision.tools.detections import Detections, BoxAnnotator

import pyglet #for sound alert

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

model = YOLO("C:/Users/admin/Downloads/AdityaPOC/safety_helmet_detection/weights/yolo8n_160epochs.pt")    # load a pretrained YOLOv8n model
model.fuse()
CLASS_NAMES_DICT = model.model.names


box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)

#box_annotator_noHelmet = BoxAnnotator(color=Color(r=255,g=0,b=0), thickness=3, text_thickness=3, text_scale=1.5)
#box_annotator_Helmet = BoxAnnotator(color=Color(r=0,g=255,b=0), thickness=3, text_thickness=3, text_scale=1.5)

def plot_bboxes(results,frame,color_code):
    xyxys = []
    confidences = []
    class_ids = []

    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        # print(class_id)
        if class_id == 1:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
        
    #Setup detections for visualization
    detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )
    
    # Format custom labels
    labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections]
    
    # Annotate and display frame
    if color_code == 'red':
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    else:
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    
    return frame


def sound_alert(frame):
        saved_nohelmetimages_counter=1
        music_voice = pyglet.resource.media('beep_alert_ok.mp3')
        music_voice.play()
        cv2.imwrite('Saved_NoHelmet/'+str(saved_nohelmetimages_counter)+'.jpg',frame)


def stream_vid(capture_index):
    
    res_count=0 
    first_alert=50
    second_alert=200
    third_alert=500
    fourth_alert=1000

    cap = cv2.VideoCapture(capture_index)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        start_time = time()
        ret, frame = cap.read()
        assert ret

        results = model(frame)
        ##
        xyxy=[]
        class_id=[]
        for result in results[0]:
            xyxy.append(result.boxes.xyxy.cpu().numpy())
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            # print(class_id)
        
        if class_id==1:
            res_count+=1
        else:
            pass
            
        if res_count == first_alert:
            sound_alert(frame)  
        elif res_count == second_alert:
            sound_alert(frame)
        elif res_count == third_alert:
            sound_alert(frame)
        elif res_count == fourth_alert:
            sound_alert(frame)
            res_count=0
        ##

        # if class_id == 4 or class_id == 5 or class_id == 0 or class_id == 3:
        if class_id == 1:

            frame =  plot_bboxes(results, frame,'red')
            cv2.putText(frame, "Warning!", (50,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

            image_encode = cv2.imencode('.jpg', frame)[1].tostring()   # convertimg resultsframe into jp and storing in db- blob
            ct = datetime.datetime.now()   #current datetimestamp
            sql = "INSERT INTO no_helmet_log (timestamp,detected,class) VALUES (%s,%s,%s)"
            val = (ct,image_encode,'NO Helmet')   
            cursor.execute(sql,val)
            db.commit()


        else:
            frame =  plot_bboxes(results, frame,'green')
            image_encode = cv2.imencode('.jpg', frame)[1].tostring()   # convertimg resultsframe into jp and storing in db- blob
            ct = datetime.datetime.now()   #current datetimestamp
            sql = "INSERT INTO no_helmet_log (timestamp,detected,class) VALUES (%s,%s,%s)"
            val = (ct,image_encode,'Helmet')   
            cursor.execute(sql,val)
            db.commit()

        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)


        cv2.putText(frame, f'FPS: {int(fps)}', (10,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_index='C:/Users/admin/Downloads/AdityaPOC/safety_helmet_detection/videos/4.mp4'
stream_vid(capture_index)