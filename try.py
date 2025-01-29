'''
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue

# Load the YOLO model and specify person class
model = YOLO("yolo11l.pt")

# Open the video file
video_path = "E:/IMG_4639.MOV"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Run YOLO tracking on the frame, tracking only persons
        results = model.track(frame, persist=True, classes=[0],conf=0.5)  # 0 is typically the person class
        
        # Check if any persons are detected
        if len(results[0].boxes) > 0:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
            # Display the annotated frame
            annotated_frame = cv2.resize(annotated_frame, (800, 600))
            cv2.imshow("YOLO Tracking - Persons", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()'''

'''from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue
from threading import Thread
import os
from datetime import datetime

# Load the YOLO model and specify person class
model = YOLO("yolo11l.pt")

# Open the video file
video_path = "E:/IMG_4639.MOV"
cap = cv2.VideoCapture(video_path)

# Initialize the queue
q2 = Queue()

# Base directory for snapshots
base_snapshot_dir = "E:/Snapshots"
os.makedirs(base_snapshot_dir, exist_ok=True)

# Store the track history
track_history = defaultdict(lambda: [])

# Frame count and interval for snapshots
frame_count = 0
snapshot_interval_frames = 30  # Save snapshots every 30 frames

def examthread(q2):
    """Thread for processing queued snapshots."""
    while True:
        # Get an item from the queue
        item = q2.get()
        if item is None:
            # Exit when stop signal is received
            break
        
        frame, results, snapshot_dir, timestamp,timestamp_str = item
                # Check if any persons are detected
        if len(results[0].boxes) > 0:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            height, width, _ = frame.shape
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                    # Convert to integers
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(width, int(x + w)), min(height, int(y + h))
                cropped_person=frame[y1:y2,x1:x2]
                cropped_person=cv2.resize(cropped_person,(200,300))
                cv2.imshow("hi",cropped_person)
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        annotated_frame = cv2.resize(annotated_frame, (800, 600))
        cv2.imshow("YOLO Tracking - Persons", annotated_frame)
                # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        q2.task_done()

# Start the snapshot processing thread
exam_thread = Thread(target=examthread, args=(q2,))
exam_thread.start()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        frame_count += 1
        
        # Run YOLO tracking on the frame, tracking only persons
        results = model.track(frame, persist=True, classes=[0], conf=0.5)  # 0 is typically the person class
        

            
            # Save snapshots at specified intervals
            
            # Generate a timestamped folder for the frame
        time = datetime.now()
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        timestamp_str=time.strftime('%Y%m%d_%H_%M_%S')
        snapshot_dir = os.path.join(base_snapshot_dir,timestamp_str)
        os.makedirs(snapshot_dir, exist_ok=True)
                
        # Queue the frame for snapshot processing
        q2.put((frame, results, snapshot_dir, timestamp,timestamp_str))
    else:
        break
q2.put(None)
exam_thread.join()
cap.release()
cv2.destroyAllWindows()
'''
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue
from threading import Thread
import os
from datetime import datetime
import mediapipe as mp
import time
q2=Queue()

base_snapshot_dir = "E:/examtesting/sus_roi_exam_solo"
os.makedirs(base_snapshot_dir, exist_ok=True)
# Load the YOLO model and specify person class
model = YOLO("yolo11n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
# Initialize Super Resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Path to the FSRCNN model
path = "FSRCNN_x3.pb"
# Read the FSRCNN model
sr.readModel(path)

# Set the model type and x3scale
sr.setModel("fsrcnn", 3)
# Open the video file
#video_path = "E:/exam.mp4"
video_path = "C:/Users/vinee/Downloads/examsolo.mp4"
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Compute vectors
    ba = a - b
    bc = c - b
    # Calculate angle using dot product and arccos
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Clip to handle precision issues
    return angle
cap = cv2.VideoCapture(video_path)
def snapshot_processor(queue):
    while True:
        item = queue.get()
        if item is None:  # Exit condition
            break

        frame, results, snapshot_dir, timestamp = item

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            if box.id is None:
                track_id = 0
            else:  
                track_id = int(box.id[0].numpy())  # Get the unique track ID

            # Ensure bounding box coordinates are within frame boundaries
            height, width, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop the detected person
            cropped_person = frame[y1:y2, x1:x2]
            sus=0
            result = sr.upsample(cropped_person)
            hand_below=0
            extreme_end=0
            slouched=0
            sharpened_image = sharpen_image(result)
            image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            image_height, image_width, _ = sharpened_image.shape
            try:
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    landmarks_array = np.array([
    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height],
    [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height],
    [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height],
    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height],
    [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height],
    [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height]
]) 
                    right_eye = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
                    left_eye=pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
                    nose =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    left_ear =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]
                    right_ear =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
                    left_shoulder=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    left_eye_outer=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER]
                    right_eye_outer=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER]
                    left_wrist=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    right_wrist=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                    left_hip=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    right_hip=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                    min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x)
                    max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x)
                    min_y = min(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y)
                    max_y = max(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y)

                    # Calculate center of the bounding box
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2

                    # Calculate width and height of the box
                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    # Add margins proportionally to both sides from the center
                    box_margin_x = 0.25 * box_width  # Adjust margin as needed
                    box_margin_y = 0.25 * box_height

                    # Calculate new bounds ensuring they're centered
                    new_min_x = max(0, center_x - (box_width/2 + box_margin_x))
                    new_max_x = min(1, center_x + (box_width/2 + box_margin_x))
                    new_min_y = max(0, center_y - (box_height/2 + box_margin_y))
                    new_max_y = min(1, center_y + (box_height/2 + box_margin_y))

                    # Convert normalized coordinates to pixel coordinates
                    min_x_px = int(new_min_x * image_width)
                    max_x_px = int(new_max_x * image_width)
                    min_y_px = int(new_min_y * image_height)
                    max_y_px = int(new_max_y * image_height)

                    # Draw the ROI on the image
                    roi_color = (0, 255, 0)  # Green for ROI box
                    cv2.rectangle(sharpened_image, (min_x_px, min_y_px), (max_x_px, max_y_px), roi_color, 2)


                    # Check if either wrist is inside the bounding box
                    left_hand_in_box = (new_min_x <= left_wrist.x <= new_max_x) and (new_min_y <= left_wrist.y <= new_max_y)
                    right_hand_in_box = (new_min_x <= right_wrist.x<= new_max_x) and (new_min_y <= right_wrist.y<= new_max_y)
                    if (not left_hand_in_box and left_wrist.visibility>=0.40 ) or (not right_hand_in_box and right_wrist.visibility>=0.40):
                        cv2.putText(sharpened_image, "Suspicious!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
                        sus=1
                    else:
                        cv2.putText(sharpened_image, "Normal", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
                    cv2.putText(sharpened_image,str(left_wrist.visibility)+" - "+str(right_wrist.visibility), (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    half=(left_shoulder.x+right_shoulder.x)/2
                    #tolerance = 0.05  # Adjust as needed based on coordinate range
                    if (left_eye_outer.x  < half): 
                        looking_left = 1
                        cv2.putText(sharpened_image, "Looking Right", (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
                    if (right_eye_outer.x > half) :
                        looking_right = 1
                        cv2.putText(sharpened_image, "Looking Left", (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
                    wristl = landmarks_array[2]
                    elbowl=landmarks_array[1]
                    elbowr=landmarks_array[4]
                    shoulderl = landmarks_array[0]
                    shoulderr = landmarks_array[3]
                    wristr = landmarks_array[5]
                    leftangle = calculate_angle(shoulderl, elbowl, wristl)
                    rightangle = calculate_angle(shoulderr, elbowr, wristr)
                    if ((wristl[1] > elbowl[1] and leftangle >= 150 and (not(left_wrist.visibility>=0.90))) or 
                        (wristr[1] > elbowr[1] and rightangle >= 150 and (not(right_wrist.visibility>=0.90)))):
                        print("hand below"+timestamp)  
                        hand_below = 1
                        cv2.putText(sharpened_image, "Hands Below", (80, 450), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
                        
                
            except:
                print("some exception occured")
            mp_drawing.draw_landmarks(
            sharpened_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Save the cropped individual image using the track ID
            snapshot_path = os.path.join(snapshot_dir, f"person_{track_id}.png")
            success = cv2.imwrite(snapshot_path, sharpened_image)
            if not success:
                print(f"Failed to save image for Track ID {track_id} at {snapshot_path}.")
            else:
                print(f"Image saved successfully for Track ID {track_id}.")

            
worker_thread = Thread(target=snapshot_processor, args=(q2,))
worker_thread.start()
frame_interval=29
frame_count=0
# Main loop using YOLO's .track()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break
    frame_count+=1
    if frame_count%frame_interval==0:
        # Run YOLO tracking on the frame, tracking only persons
        results = model.track(frame, persist=True, classes=[0], conf=0.5)

        # Create a timestamp folder for the current frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(base_snapshot_dir, timestamp)
        os.makedirs(snapshot_dir, exist_ok=True)

        # Add the frame and results to the processing queue
        q2.put((frame, results, snapshot_dir, timestamp))

cap.release()
cv2.destroyAllWindows()
q2.put(None)




