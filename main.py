import numpy as np
import torch
import cvzone
from super_gradients.training import models
import cv2
import math
from sort import *

vid_link = r'C:\Users\Admin\Desktop\vd/video2.mp4'
cap = cv2.VideoCapture(vid_link)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)

zone = np.array([[500, 200], [500, 540], [800, 540], [800, 200]], np.int32)

tracker = Sort()
classnames = []

with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

while cap.isOpened():
    detections = np.empty((0, 5))
    people_count = 0
    total_people_in_frame = []

    rt, video = cap.read()
    video = cv2.resize(video, (1080, 740))

    result = model.predict(video, conf=0.50)[0]
    bboxs = result.prediction.bboxes_xyxy
    confidence = result.prediction.confidence
    labels = result.prediction.labels

    for (bboxs, confidence, labels) in zip(bboxs, confidence, labels):
        x1, y1, x2, y2 = np.array((bboxs))
        x1, y1, x2, y2 = int(bboxs[0]), int(bboxs[1]), int(bboxs[2]), int(bboxs[3])
        confidence = math.ceil(confidence * 100)
        labels = int(labels)
        classdetect = classnames[labels]
        w, h = x2 - x1, y2 - y1

        if classdetect == 'person':
            new_detections = np.array([x1, y1, x2, y2, confidence])
            detections = np.vstack((detections, new_detections))


    track_result = tracker.update(detections)
    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        total_people_in_frame.append(id)

        cv2.circle(video, (cx, cy), 6, (0, 255, 255), -1)
        cv2.rectangle(video, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(video, f'{id}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        cv2.polylines(video,[zone],True,(0,0,255),4)
        counts = cv2.pointPolygonTest(zone, pt=(cx, cy), measureDist=False)
        if counts == 1:  # Inside the defined region
            people_count += 1

    # Display the count of people in the region on the frame
    cvzone.putTextRect(video,f'People in Region: {people_count}', (0, 32),scale=2)
    cvzone.putTextRect(video, f'Total People Detected: {len(total_people_in_frame)}', (600, 32),scale=2)

    cv2.imshow('frame', video)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()