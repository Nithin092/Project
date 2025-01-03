import cv2 # OpenCV - computer vision module
import numpy as np # Numpy - large matrix and arithmetic operations
import torch # PyTorch - Deep Learning Framework
import time 

from temp.temperature import get_temperature

import firebase_admin # Firebase Realtime DB Client Module
from firebase_admin import credentials, db

cred = credentials.Certificate("pythonjs.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dream-602dc-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

API_KEY = "810366b35b463049ebc14f6b796cbbd7"  # Replace with your actual API key
city = "Idukki"  # Replace with your desired city
temperature = get_temperature(API_KEY, city)  # Call the function

if temperature is not None:
    print(f"The current temperature in {city} is {temperature}°C.")
else:
    print("Could not retrieve the temperature.")


# function to send data to firebase

# def send_data_to_firebase(detections):
#     detections_ref = db.reference('/detections')
#     detections_ref.set({
#         'counts': detections
#     })

def send_data_to_firebase(detections, temperature):
    """
    Sends the grid detection counts and temperature to Firebase.
    :param detections: List of counts for each grid section
    :param temperature: Current temperature value
    """
    detections_ref = db.reference('/detections')
    detections_ref.set({
        'counts': detections,
        'temperature': temperature
    })

# function to calculate number of people in sections

def calculate_detections_in_polygons(labels, cords, polygons, width, height):
        detections_in_polygons = [0] * len(polygons)

        for i, (label, cord) in enumerate(zip(labels, cords)):
            if int(label) == 0:
                x_center = int((cord[0]*width + cord[2]*width) / 2)
                y_center = int((cord[1]*height + cord[3]*height) / 2)
                
                for j, poly in enumerate(polygons):
                    if cv2.pointPolygonTest(poly, (x_center, y_center), False) >= 0:
                        detections_in_polygons[j] += 1
        print(detections_in_polygons)
        return detections_in_polygons

# YOLOv5 Small - Person Detection Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = model.to(device)

prev_time = 0
last_update_time = time.time()

def rotate_point(x, y, cx, cy, angle):

    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    x -= cx
    y -= cy
    
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    
    x = x_new + cx
    y = y_new + cy
    return int(x), int(y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    height, width = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model([frame_rgb])

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # PARTITIONING CODE

    for i, (label, cord) in enumerate(zip(labels, cords)):
        if int(label) == 0:
            x1, y1, x2, y2, conf = int(cord[0]*width), int(cord[1]*height), int(cord[2]*width), int(cord[3]*height), cord[4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    x1, x2 = 1 * width // 4, 3 * width // 4
    y1 = 2 * height // 3
    y2 = 5 * height // 12

    angle = 65 
    x1_start_from_y2, _ = rotate_point(x1, y1, x1, y2, -angle)
    x1_rotated_bottom, _ = rotate_point(x1, height, x1, y1, angle)
    x2_start_from_y2, _ = rotate_point(x2, y1, x2, y2, angle)
    x2_rotated_bottom, _ = rotate_point(x2, height, x2, y1, -angle)

    cv2.line(frame, (x1_start_from_y2, y2), (x1, y1), (255, 0, 0), 1)
    cv2.line(frame, (x1, y1), (x1_rotated_bottom, height), (255, 0, 0), 1)
    cv2.line(frame, (x2_start_from_y2, y2), (x2, y1), (255, 0, 0), 1)
    cv2.line(frame, (x2, y1), (x2_rotated_bottom, height), (255, 0, 0), 1)
    cv2.line(frame, (0, y1), (width, y1), (255, 0, 0), 1)
    cv2.line(frame, (0, y2), (width, y2), (255, 0, 0), 1)

    polygons = [
    np.array([[x1_start_from_y2, y2], [x1, y1], [0, y1], [0, y2]], np.int32),  # TL
    np.array([[x1_start_from_y2, y2], [x1, y1], [x2, y1], [x2_start_from_y2, y2]], np.int32),  # TM
    np.array([[x2_start_from_y2, y2], [x2, y1], [width, y1], [width, y2]], np.int32),  # TR
    np.array([[x1, y1], [x1_rotated_bottom, height], [0, height], [0, y1]], np.int32),  # BL
    np.array([[x1, y1], [x1_rotated_bottom, height], [x2_rotated_bottom, height], [x2, y1]], np.int32),  # BM
    np.array([[x2, y1], [x2_rotated_bottom, height], [width, height], [width, y1]], np.int32)  # BR
    ]

    # PARTITIONING ENDS

    # Checker for every 5 seconds

    if current_time - last_update_time >= 5:
        send_data_to_firebase(calculate_detections_in_polygons(labels, cords, polygons, width, height),temperature)
        last_update_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Classroom Sections', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
send_data_to_firebase([0,0,0,0,0,0])