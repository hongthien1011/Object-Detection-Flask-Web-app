import cv2
import requests
import numpy as np
# import matplotlib.pyplot as plt
from Classifine_Noise import PrePocessing
from PIL import Image as im
import os

def onetotwoimage(path):
    PROCSESS_FOLDER = r'./static/display'
    # image1 = cv2.imread(image_path1)
    image1, image2 = PrePocessing(path)
    data1 = im.fromarray(image1)
    data1.save(os.path.join(PROCSESS_FOLDER,"display.jpg"))
    data2 = im.fromarray(image2)
    data2.save(os.path.join(PROCSESS_FOLDER,"preprocessed.jpg"))
    image1 = cv2.imread(os.path.join(PROCSESS_FOLDER,"display.jpg"))
    image2 = cv2.imread(os.path.join(PROCSESS_FOLDER,"preprocessed.jpg"))
    return image1, image2

# Apply NMS to response_data1 and response_data2 bounding boxes
def apply_nms(annotations, overlap_threshold=0.9):
    # Convert bounding boxes to [x1, y1, x2, y2] format
    boxes = np.array([(ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']) for ann in annotations])
    scores = np.array([ann['confidence'] for ann in annotations])

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, overlap_threshold)
    
    selected_annotations = [annotations[i] for i in indices]
    return selected_annotations

def getjson(image1,image2,url= 'http://10.182.220.134:8000/detection/img_object_detection_to_json'):
    # Convert the image to base64
    _, img_encoded = cv2.imencode('.jpg', image1)
    image_base641 = img_encoded.tobytes()
    # image_base64
    # Convert the image to base64
    _, img_encoded = cv2.imencode('.jpg', image2)
    image_base642 = img_encoded.tobytes()
    # Create a dictionary for the files parameter
    files1 = {'file': ('image.jpg', image_base641)}
    # Create a dictionary for the files parameter
    files2 = {'file': ('image.jpg', image_base642)}
    # Send a POST request to the API for object detection
    response1 = requests.post(url, files=files1)
    # Send a POST request to the API for object detection
    response2 = requests.post(url, files=files2)
    # Parse the response JSON
    response_data1 = response1.json()
    response_data2 = response2.json()
    # Calculate scaling factors
    original_width1, original_height1 = image1.shape[1], image1.shape[0]
    resized_width2, resized_height2 = image2.shape[1], image2.shape[0]

    width_scale = original_width1 / resized_width2
    height_scale = original_height1 / resized_height2
        # Scale bounding box coordinates in response_data2
    for annotation in response_data2['detect_objects']:
        annotation['xmin'] *= width_scale
        annotation['xmax'] *= width_scale
        annotation['ymin'] *= height_scale
        annotation['ymax'] *= height_scale

    # Combine response_data1 and response_data2 (scaled)
    combined_response_data = {
        'detect_objects': response_data1['detect_objects'] + response_data2['detect_objects'],
        'detect_objects_names': response_data1['detect_objects_names'] + response_data2['detect_objects_names']
    }
    nms_annotations1 = apply_nms(combined_response_data['detect_objects'])
    nms_annotation_names1 = [combined_response_data['detect_objects_names'][combined_response_data['detect_objects'].index(annotation)] for annotation in nms_annotations1]
    nms_data1 = {
        'detect_objects': nms_annotations1,
        'detect_objects_names': nms_annotation_names1
    }
    return nms_data1



def better_detect(path):
    image1, image2 = onetotwoimage(path)
    response = getjson(image1, image2)
    return response

def better_detect_on_preprocessed_image(image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    response = getjson(image1, image2)
    return response
    
    