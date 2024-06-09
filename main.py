import numpy as np
import torch
import os
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort as DeepSortTracker
from ultralytics import YOLO

""" Initiate Cuda, if You have it """
print('GPU Available : ', torch.cuda.is_available())
print('GPU Device    : ', torch.cuda.get_device_name())

""" Load Model Yolov8 """
model = YOLO("yolov8x.pt")
model.to('cuda')

""" Load Deep Sort Tracker """
object_tracker = DeepSortTracker()

""" Initialize ROI and ROI set for Object Counting """
# The ROI coordinates are selected by inspect the screenshot of the video
ROI_cords = [(618, 361), (491, 417), (701, 430), (800, 370)]
ROI_counter_id = set()

""" Load Video """
video_path = os.path.join('.', 'data', 'AbbeyRoad.mp4')
cap = cv2.VideoCapture(video_path)

""" Read Frames """
while cap.isOpened():
    ret, frame = cap.read()

    """ Detect Object """
    # use classes parameter to only detect human
    results = model(frame, classes=[0])

    # Extract Yolov8 Output
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x, y, w, h, prob, class_id = r
            int(class_id)
            detections.append([(x, y, w, h), prob, class_id])

        # Track Object
        trackers = object_tracker.update_tracks(detections, frame=frame)

        for track in trackers:
            if not track.is_confirmed():
                continue
            if track.original_ltwh is None or track.track_id is None:
                continue
            bbox = track.original_ltwh
            track_id = track.track_id

            x1, y1, x2, y2 = bbox

            # Object Counting
            crossing_region_lb = cv2.pointPolygonTest(np.array(ROI_cords), (int(x1), int(y2)), False)
            crossing_region_rb = cv2.pointPolygonTest(np.array(ROI_cords), (int(x2), int(y2)), False)
            if crossing_region_lb > 0 or crossing_region_rb > 0:
                ROI_counter_id.add(track_id)

            # Drawing ROI shape
            cv2.line(frame, (ROI_cords[0]), ROI_cords[1], (0, 255, 0), 3)
            cv2.line(frame, (ROI_cords[1]), ROI_cords[2], (0, 255, 0), 3)
            cv2.line(frame, (ROI_cords[2]), ROI_cords[3], (0, 255, 0), 3)
            cv2.line(frame, (ROI_cords[3]), ROI_cords[0], (0, 255, 0), 3)

            # Drawing Counter Box
            counter_box_cords = [(0, 0), (380, 80)]
            cv2.rectangle(frame, (counter_box_cords[0]), (counter_box_cords[1]), (255, 0, 0), -1)

            # Drawing bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)

            cv2.putText(frame, "Crossed: " + str(len(ROI_counter_id)), (0, 0 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), thickness=2)

    # Show Video
    cv2.imshow("Abbey Road Crossing", frame)

    # Video will be closed when 'q' button is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
