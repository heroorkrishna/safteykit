# import cv2
# import numpy as np

# # Load the annotations file
# annotations = np.load("C:/Users/admin/OneDrive/Documents/safteykitdetection/ppedet.tfrecord")

# # Load the detected images from the folder
# folder_images = ["C:/Users/admin/OneDrive/Documents/safteykitdetection/outputimages/20230412-104828428804.jpg"]

# # Define the detection threshold for overlap with ground truth objects
# threshold = 0.5

# # Initialize counters for true positive detections and ground truth objects
# true_positives = 0
# ground_truth_objects = 0

# # Loop over the images in the folder
# for image_path in folder_images:
#     # Load the image from the folder
#     folder_image = cv2.imread(image_path)

#     # Extract the filename from the image path
#     filename = image_path.split("/")[-1]

#     # Find the corresponding annotations for the image
#     image_annotations = annotations[annotations[:, 0] == filename]

#     # Loop over the annotations for the image
#     for annotation in image_annotations:
#         # Extract the class ID and bounding box coordinates
#         class_id = annotation[1]
#         bbox = annotation[2:]

#         # Detect objects in the image using your object detection algorithm
#         # and extract the detected bounding boxes and class IDs
#         detected_boxes = []
#         detected_class_ids = []

#         # Calculate the intersection-over-union (IoU) between the detected boxes
#         # and the ground truth boxes for the current annotation
#         iou_scores = []
#         for detected_box, detected_class_id in zip(detected_boxes, detected_class_ids):
#             iou = calculate_iou(detected_box, bbox)
#             iou_scores.append(iou)

#         # Check if there is a true positive detection for the current annotation
#         if len(iou_scores) > 0 and max(iou_scores) > threshold:
#             true_positives += 1

#         # Increment the ground truth object counter
#         ground_truth_objects += 1

# # Calculate the true positive rate
# tpr = true_positives / ground_truth_objects

# print("True positive rate:", tpr)

import os
import numpy as np
import tensorflow as tf

out_dir = '../out/'

frame_lvl_record =  'validate00.tfrecord'


for example in tf.python_io.tf_record_iterator(frame_lvl_record):

    dataset = dict()
    
    tf_example = tf.train.Example.FromString(example)
    tf_seq_example = tf.train.SequenceExample.FromString(example)

    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

    vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
    vid_labels = tf_example.features.feature['labels'].int64_list.value    

    rgb_frame = []
    audio_frame = []
    
    sess = tf.InteractiveSession()

    for i in range(n_frames):
        rgb_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['rgb']
            .feature[i].bytes_list.value[0], tf.uint8)
                                 , tf.float32).eval())
        
        audio_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['audio']
            .feature[i].bytes_list.value[0], tf.uint8)
                                   , tf.float32).eval())

    sess.close()
    
    dataset['id'] = vid_id
    dataset['labels'] = list(vid_labels)
    dataset['rgb_frame'] = list(rgb_frame)
    dataset['audio_frame'] = list(audio_frame)
            
    np.save(out_dir + vid_id + '.npy', np.array(dataset))
    
    break