#TEST

from ultralytics import YOLO
import torch
import cv2
import time
from time import time
import numpy as np
import json
from IPython.display import clear_output
import psycopg2
import datetime
import os

model = YOLO('C:/Users/admin/OneDrive/Documents/safteykitdetection/models/yolos_250epochs.pt')
model.fuse()
CLASS_NAMES_DICT = model.model.names 
print(CLASS_NAMES_DICT)

# db connection
conn = psycopg2.connect("dbname=safteykitdetection user=postgres password=meonly")

# Create a cursor object
cur = conn.cursor()


color_codes={0:(255,96,208), 1:(160,32,255), 2:(80,208,255), 3:(96,255,128), 4:(255,160,16), 5:(0,32,255), 6:(160,128,96), 7:(255,208,160)}
warning_msg_region={0:(45,180), 4:(45,220), 5:(45,260), 6:(45,300), 7:(45,340),8:(45,370)}
conf_threshold=0.4  
fps_per_sec=20
latency_between_frames=5
shoes_id=7
line_crossed = 8
non_safety_ids=[0,4,5,6,shoes_id]
ald=0 #adj - adjust line detection

#read line coordibnates from file
filepath = 'line_coordinates.data'
with open(filepath) as fp:
   lines = json.loads(fp.readline())
   fp.close()
line1=lines['line1']
line2=lines['line2']


def create_alert(cls_id,orig_images,shoes_xyxy):
    non_safety_idss=non_safety_ids[0:-1]
    if cls_id in non_safety_idss or cls_id == shoes_id:
        warning_message="Warning!..{} detected".format(CLASS_NAMES_DICT[cls_id])
        cv2.putText(orig_images, warning_message, warning_msg_region[cls_id], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # image_encode = cv2.imencode('.jpg', orig_images)[1].tostring()   # convertimg resultsframe into jp and storing in db- blob
        # image_encode = cv2.imencode('.jpg', orig_images)[1].tobytes()
        # Encode image as JPEG and store as binary data in a bytes object
        # _, image_bytes = cv2.imencode('.jpg', orig_images)
        # Encode image as JPEG and store in folder
        img_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')}.jpg"
        img_path = os.path.join("C:/Users/admin/OneDrive/Documents/safteykitdetection/outputimages", img_name)
        cv2.imwrite(img_path, orig_images)
        # Store image as binary data in PostgreSQL table
        with open(img_path, "rb") as f:
            image_data = f.read()
        # print(image_encode)
        ct = datetime.datetime.now()   #current datetimestamp
        sql = "INSERT INTO safteykitdetectionlogs (timestamp,image,class) VALUES (%s,%s,%s)"
        val = (ct,psycopg2.Binary(image_data),'{}'.format(CLASS_NAMES_DICT[cls_id]))   
        cur.execute(sql,val)
        conn.commit()

        if len(shoes_xyxy) != 0:
            orig_images=detect_person_crossline(orig_images,shoes_xyxy)
        else:
            pass
    else:
        pass
    return orig_images


def detect_person_crossline(frame,shoe_boxes):
    for shoe_box in shoe_boxes:
        if shoe_box[0] > line1[0][0]+ald and shoe_box[2] < line1[1][0] and shoe_box[1] > line1[0][1]+ald and shoe_box[3] < line1[1][1]:
            cv2.putText(frame, "Warning! Person crossed the Yellow Line", warning_msg_region[line_crossed], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print('line cross detcted')
        elif shoe_box[0] > line2[0][0]+ald and shoe_box[2] < line2[1][0] and shoe_box[1] > line2[0][1]+ald and shoe_box[3] < line2[1][1]:
            cv2.putText(frame, "Warning! Person crossed the Yellow Line", warning_msg_region[line_crossed], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print('line cross detcted')
        else:
            pass
        return frame 

#Logic Referrence
#   (X1 > x1) & (X2 < X2) &(Y1 > y1) & (Y2 < Y2)
#   xyxy=[12,323,343,322]
#   line1 = [[622, 283], [421, 719]]
#   line2 = [[422, 253], [221, 119]]
      
    
def process_detections(results,counter):
     for result in results:
        xyxys=result.boxes.xyxy.cpu().numpy().astype(int)
        classids=result.boxes.cls.cpu().numpy().astype(int)
        confs=result.boxes.conf.cpu().numpy().astype(float)
        orig_images=result.orig_img
        shoes_index_pos=np.where(classids==shoes_id)    
        shoe_bbox=xyxys[shoes_index_pos]

     if len(confs) != 0:   
         for i in range(0,len(confs),1):
             xyxy=xyxys[i]
             cls_id=classids[i]
             conf=confs[i]
             if conf > conf_threshold:#and cls_id in non_safety_ids:  ##remove remove # to exclude detection safety equips
                res=str(CLASS_NAMES_DICT[cls_id])
                frame_color=color_codes[cls_id]
                cv2.rectangle(orig_images, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), frame_color, 1) 
                cv2.putText(orig_images,res, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 1)
                orig_images=create_alert(cls_id,orig_images,shoe_bbox)
             else:
                 pass      
     else:
          pass  
     return orig_images

def stream_vid(capture_index,line1xy,line2xy):    
    counter=0
    video = cv2.VideoCapture(capture_index)
    assert video.isOpened()
    video.set(cv2.CAP_PROP_FPS, fps_per_sec)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    while True:
        start_time = time()
        ret, frame = video.read()
        assert ret
        
        results = model(frame)
        counter=0 if counter == 2000 else counter+1
        frame =  process_detections(results,counter)
        
        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)

        cv2.putText(frame, f'FPS: {int(fps)}', (10,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        cv2.line(frame, (line1xy[0]), (line1xy[1]), 255, 2)
        cv2.line(frame, (line2xy[0]), (line2xy[1]), 255, 2)
        cv2.imshow('YOLOv8 Detection', frame)
        #clear_output(wait=True)
        if cv2.waitKey(latency_between_frames) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()



capture_index='C:/Users/admin/OneDrive/Documents/safteykitdetection/videos/video_tmp9.mp4'
stream_vid(capture_index,line1,line2)
