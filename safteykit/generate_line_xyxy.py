import cv2
import json
def create_single_frame(video_name):
  cap = cv2.VideoCapture(video_name)
  if not cap.isOpened():
      print("Error opening video file")
      exit()
  ret, frame = cap.read()
  if not ret:
      print("Error reading frame")
      exit()
  cv2.imwrite('temp_image.jpg', frame)
  cap.release()

cords1=[]
cords2=[]

def click_event1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pots=(x,y)
        cords1.append(pots)
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pots=(x,y)
        cords2.append(pots)

def draw_line(capture_index):
  number_of_lines=2

  create_single_frame(capture_index)
  img = cv2.imread('temp_image.jpg')
  cv2.imshow('image', img)
  cv2.setMouseCallback('image', click_event1)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  print((cords1[0]), (cords1[1]))

  cv2.imshow('image', img)
  cv2.setMouseCallback('image', click_event2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  print((cords2[0]), (cords2[1]))

  cv2.line(img, (cords1[0]), (cords1[1]), 255, 5)
  cv2.line(img, (cords2[0]), (cords2[1]), 255, 5)
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  content={'line1':cords1,'line2':cords2}
  with open('line_coordinates.data', 'w') as f:
     f.write(json.dumps(content))
     f.close()

  print('done')
