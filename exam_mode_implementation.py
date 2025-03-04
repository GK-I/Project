from flask import Flask, render_template, Response, jsonify,redirect,url_for
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue
from threading import Thread
import os
from datetime import datetime,timedelta,timezone
import mediapipe as mp
import time
from sqlalchemy import func
from flask_sqlalchemy import SQLAlchemy
from flask import render_template
import gc
import math
import base64
from flask_socketio import SocketIO
from mtcnn.mtcnn import MTCNN
import traceback
q2 = Queue()
detector=MTCNN()
#video_path="C:/Users/vinee/Downloads/examnew.mp4"
# Initialize global variables

base_snapshot_dir = "E:/dont_mind/examnew_results"
os.makedirs(base_snapshot_dir, exist_ok=True)
#socketio = SocketIO(app,max_http_buffer_size=100 * 1024 * 1024,cors_allowed_origins="*")
# Initialize models and utilities
model = YOLO("yolo11n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0,refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Initialize Super Resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("D:/FinalProject/MediaPipeLearn/FSRCNN/FSRCNN-small_x3.pb")
sr.setModel("fsrcnn", 3)
track_img={}
def face(image_rgb,sharpened_image):
    results_face = face_mesh.process(image_rgb)

    # Convert the image back to BGR
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = sharpened_image.shape
    face_3d = []
    face_2d = []
    x, y = 0, 0
    p1, p2 = (0, 0), (0, 0)
    face_landmarks = None
    if results_face.multi_face_landmarks:
        landmarks_detected=True
        for face_landmarks in results_face.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
                            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 3 * img_w
            cam_matrix = np.array([
                                [focal_length, 0, 3*img_h / 2],
                                [0, focal_length, 3*img_w / 2],
                                [0, 0, 3]
                            ])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix,flags=cv2.SOLVEPNP_UPNP)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360  # Negating the y angle to correct the direction
            z = angles[2] * 360

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    else:
        landmarks_detected=False
    return(x,y,p1,p2,face_landmarks,landmarks_detected)

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

'''def calculate_angle_1(nose, left_eye, right_eye):
    # Find face center
    face_center_x = (left_eye.x + right_eye.x) / 2
    face_center_y = (left_eye.y + right_eye.y) / 2
    
    # Compute angle (in degrees) between nose and face center
    delta_x = nose.x - face_center_x
    delta_y = nose.y - face_center_y
    angle = math.degrees(math.atan2(delta_y, delta_x))  # Convert radians to degrees
    
    return angle'''
stop_signal_received_exam=False
def received_stop_signal(stop_signal_received):
    global stop_signal_received_exam
    print("received")
    stop_signal_received_exam=stop_signal_received

def exam_processor():
    global stop_signal_received_exam
    sus_tracker={}
    print("started.................................")
    from app import send_suspicious_frame,send_last_frame
    while True:
        item = q2.get()
        if item is None:
            break
            
        frame, results, snapshot_dir,frame_count,timestamp ,is_last_frame= item
        if stop_signal_received_exam==True:
            is_last_frame=1
            print(".......................IS LAST FRAME SET.....................")
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            track_id = int(box.id[0].numpy()) if box.id is not None else 0
            
            # Crop and process person
            height, width, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            cropped_person = frame[y1:y2, x1:x2]
            looking_left=0
            looking_right=0
            hand_below=0
            sus_flag=0
            looking_right_flag=0
            looking_left_flag=0
            hand_below_flag=0
            # Process image
            result = sr.upsample(cropped_person)
            del cropped_person
            gc.collect()
            sharpened_image = sharpen_image(result)
            copy=sharpened_image.copy()
            del result
            gc.collect()
            image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                image_height, image_width, _ = sharpened_image.shape
                
                # Process landmarks for detection
                try:
                    # Extract key landmarks   
                    left_shoulder=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]                 
                    half_x=(left_shoulder.x+right_shoulder.x)/2
                    half_y=(left_shoulder.y+right_shoulder.y)/2
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
     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height],
    [landmarks[mp_pose.PoseLandmark.LEFT_EYE].x*image_width,
      landmarks[mp_pose.PoseLandmark.LEFT_EYE].y*image_height],
    [half_x*image_width,half_y*image_height],
    [landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x*image_width,
    landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y*image_height]])
                    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                    right_eye = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
                    left_eye=pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
                    nose =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    left_ear =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]
                    right_ear =pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]

                    left_eye_inner=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER]
                    right_eye_inner=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER]
                    left_eye_outer=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER]
                    right_eye_outer=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER]
                    left_wrist=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    right_wrist=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                    left_hip=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    right_hip=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                    left_elbow=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
                    right_elbow=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
                    # Calculate ROI


                    #cv2.putText(sharpened_image,str(left_wrist.visibility)+" - "+str(right_wrist.visibility), (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    #tolerance = 0.05  # Adjust as needed based on coordinate range

                    wristl = landmarks_array[2]
                    elbowl=landmarks_array[1]
                    elbowr=landmarks_array[4]
                    shoulderl = landmarks_array[0]
                    shoulderr = landmarks_array[3]
                    wristr = landmarks_array[5]
                    leftangle = calculate_angle(shoulderl, elbowl, wristl)
                    rightangle = calculate_angle(shoulderr, elbowr, wristr)
                    cv2.putText(sharpened_image, str(round(leftangle,2)), (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.putText(sharpened_image, str(round(rightangle,2)), (80, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    text=str(str(left_wrist.visibility)+"<==>"+str(right_wrist.visibility))
                    cv2.putText(sharpened_image, text, (30, 1500),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    (x,y,p1,p2,face_landmarks,landmarks_detected)=face(image_rgb,sharpened_image)
                    if landmarks_detected==True:
                            
                        cv2.line(sharpened_image, p1, p2, (255, 0, 0), 3)

                        mp_drawing.draw_landmarks(
                                sharpened_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        cv2.putText(sharpened_image, "yaw = "+str(round(y,2)), (80, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        cv2.putText(sharpened_image, "pitch = "+str(round(x,2)), (80, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    elif landmarks_detected==False:
                        faces = detector.detect_faces(sharpened_image)
                        if not faces:
                            print("")
                            landmarks_detected=False
                        else:
                            largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
                            x, y, w, h = largest_face['box']

                            # Extract and preprocess the face
                            face_img = sharpened_image[y:y + h, x:x + w]
                            face_img = cv2.resize(face_img, (160, 160))
                            face_image_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            (x,y,p1,p2,face_landmarks,landmarks_detected)=face(face_image_rgb,sharpened_image)
                            if landmarks_detected==True:
                            
                                cv2.line(sharpened_image, p1, p2, (255, 0, 0), 3)

                                    # Add the text on the image
                                    #cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                                    #cv2.putText(image, f"x: {np.round(x,2)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    #cv2.putText(image, f"y: {np.round(y,2)}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    #cv2.putText(image, f"z: {np.round(z,2)}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                    # Draw face mesh
                                mp_drawing.draw_landmarks(
                                        sharpened_image,
                                        landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                                cv2.putText(sharpened_image, "yaw = "+str(round(y,2)), (80, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                                cv2.putText(sharpened_image, "pitch = "+str(round(x,2)), (80, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    '''min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,(left_elbow.x+left_wrist.x)/2,(right_elbow.x+right_wrist.x)/2)
                    max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,(right_elbow.x+right_wrist.x)/2,(left_elbow.x+left_wrist.x)/2)
                    min_y = min(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,(left_elbow.y+left_wrist.y)/2,(right_elbow.y+right_wrist.y)/2)
                    max_y = max(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,(right_elbow.y+right_wrist.y)/2,(left_elbow.y+left_wrist.y)/2)'''
                    '''if x>-9 and x<-5.5 and y<-2 and y>-5:
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x)+0.15
                    else:
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x)
                    if x>-9 and x<-5.5 and y>2 and y<5:
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x)-0.15
                    else:
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x)
                    min_y = min(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,left_elbow.y,right_elbow.y)
                    max_y = max(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,right_elbow.y,left_elbow.y)'''
                    if landmarks_detected==False:  # Default bounding box
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x)
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x)
                    elif -9 < x < -5.5 and -5 < y < 0 and landmarks_detected==True:  # Head tilted down and turned right
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x) - 0.1  # Move left boundary right
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x) + 0.1  # Move right boundary right

                    elif -9 < x < -5.5 and 0 < y < 5 and landmarks_detected==True:  # Head tilted down and turned left
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x) - 0.1  # Move left boundary left
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x) + 0.1  # Move right boundary left

                    else:
                        min_x = min(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,left_elbow.x,right_elbow.x)
                        max_x = max(left_shoulder.x, left_ear.x, right_ear.x, right_shoulder.x, left_hip.x, right_hip.x,right_elbow.x,left_elbow.x)
                    min_y = min(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,left_elbow.y,right_elbow.y)
                    max_y = max(left_shoulder.y, left_ear.y, right_ear.y, right_shoulder.y, left_hip.y, right_hip.y,right_elbow.y,left_elbow.y)
                    
                    # Calculate center of the bounding box
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2

                    # Calculate width and height of the box
                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    # Add margins proportionally to both sides from the center
                    #box_margin_x = 0.20 * box_width  
                    box_margin_x= 0.25 * box_height
                    box_margin_y = 0.25 * box_height
                    #box_margin_y =1.0
                    # Calculate new bounds ensuring they're centered
                    new_min_x = max(0, center_x - (box_width/2 + box_margin_x))
                    new_max_x = min(1, center_x + (box_width/2 + box_margin_x))
                    new_min_y = max(0, center_y - (box_height/2 + box_margin_y))
                    new_max_y = 1.5
                    #print(round(left_wrist.y,2),"<-->",round(right_wrist.y,2))
                    # Convert normalized coordinates to pixel coordinates
                    min_x_px = int(new_min_x * image_width)
                    max_x_px = int(new_max_x * image_width)
                    min_y_px = int(new_min_y * image_height)
                    max_y_px = int(new_max_y * image_height)
                    mp_drawing.draw_landmarks(
                        sharpened_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    # Draw the ROI on the image
                    roi_color = (0, 255, 0)  # Green for ROI box
                    cv2.rectangle(sharpened_image, (min_x_px, min_y_px), (max_x_px, max_y_px), roi_color, 2)
                    
                    # Check for suspicious behavior
                    left_hand_in_box = (new_min_x <= left_wrist.x <= new_max_x) and (new_min_y <= left_wrist.y <= new_max_y)
                    right_hand_in_box = (new_min_x <= right_wrist.x <= new_max_x) and (new_min_y <= right_wrist.y <= new_max_y)
                    
                    if (not left_hand_in_box and left_wrist.visibility >= 0.50) or (not right_hand_in_box and right_wrist.visibility >= 0.50):
                        cv2.putText(sharpened_image, "Suspicious!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        print("_____________suspicious posture___________________",track_id)
                        sus_dir = os.path.join("E:/dont_mind/sus_v_only", timestamp.strftime("%Y%m%d_%H%M%S"))
                        os.makedirs(sus_dir, exist_ok=True)
                        sus_path = os.path.join(sus_dir, f"person_{track_id}.png")
                        cv2.imwrite(sus_path, sharpened_image)
                        tex="Suspicious posture"
                        sus_flag=1
                        send_suspicious_frame(copy,tex)
                        continue
                    if landmarks_detected==True:
                        if ((wristl[1] > elbowl[1] and leftangle >= 160 and (x<-9)  ) or 
                            (wristr[1] > elbowr[1] and rightangle >= 160 and (x<-9))):
                            print(track_id,"hand below")  
                            cv2.putText(sharpened_image, f"Hand below", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                            hand_below = 1
                            hand_dir = os.path.join("E:/dont_mind/hand_below_only_12degree", timestamp.strftime("%Y%m%d_%H%M%S"))
                            os.makedirs(hand_dir, exist_ok=True)
                            hand_path = os.path.join(hand_dir, f"person_{track_id}.png")
                            cv2.imwrite(hand_path, sharpened_image)


                    half=(left_shoulder.x+right_shoulder.x)/2
                    l=(left_eye_outer.x+left_ear.x)/ 2
                    r=(right_eye_outer.x+right_ear.x)/ 2

                    if landmarks_detected==True:
                        if (y<-5 and x>-5.5):
                            looking_right=1
                            cv2.putText(sharpened_image, "Looking right", (80, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                            right_dir = os.path.join("E:/dont_mind/right", timestamp.strftime("%Y%m%d_%H%M%S"))
                            os.makedirs(right_dir, exist_ok=True)
                            right_path = os.path.join(right_dir, f"person_{track_id}.png")
                            cv2.imwrite(right_path, sharpened_image)
                        elif ((y>5 and x>-5.5)):
                            looking_left=1
                            cv2.putText(sharpened_image, "Looking left", (80, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                            left_dir = os.path.join("E:/dont_mind/left", timestamp.strftime("%Y%m%d_%H%M%S"))
                            os.makedirs(left_dir, exist_ok=True)
                            
                            left_path = os.path.join(left_dir, f"person_{track_id}.png")
                            cv2.imwrite(left_path, sharpened_image)
                    elif(landmarks_detected==False and l < half):
                        looking_right=1
                        cv2.putText(sharpened_image, "Looking right", (80, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        right_dir = os.path.join("E:/dont_mind/right", timestamp.strftime("%Y%m%d_%H%M%S"))
                        os.makedirs(right_dir, exist_ok=True)

                        
                        right_path = os.path.join(right_dir, f"person_{track_id}.png")
                        cv2.imwrite(right_path, sharpened_image)
                    elif(landmarks_detected==False and r > half):
                        looking_left=1
                        cv2.putText(sharpened_image, "Looking left", (80, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        left_dir = os.path.join("E:/dont_mind/left", timestamp.strftime("%Y%m%d_%H%M%S"))
                        os.makedirs(left_dir, exist_ok=True)
                        
                        left_path = os.path.join(left_dir, f"person_{track_id}.png")
                        cv2.imwrite(left_path, sharpened_image)
                    # Draw pose landmarks
                    #mp_drawing.draw_landmarks(sharpened_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    if track_id not in sus_tracker:
                        sus_tracker[track_id]={"hand_below":0,"looking_left":0,"looking_right":0}
                    else:
                        if looking_left:
                            sus_tracker[track_id]["looking_left"]+=1
                        elif looking_right:
                            sus_tracker[track_id]["looking_right"]+=1
                        elif hand_below:
                            sus_tracker[track_id]["hand_below"]+=1
                    print("data added")
                    #print(sus_tracker)
                    if track_id not in track_img:
                        track_img[track_id]={"hand_below":None,"looking_left":None,"looking_right":None}
                    elif landmarks_detected==True:
                        if looking_left:
                            track_img[track_id]["looking_left"]=copy
                        elif looking_right:
                            track_img[track_id]["looking_right"]=copy
                        if hand_below:
                            track_img[track_id]["hand_below"]=copy                       
                        if sus_flag==1:
                            print("sus")
                        if  (frame_count%1800==0 or is_last_frame):
                            hand_below_duration=sus_tracker[track_id]['hand_below']
                            left_count=sus_tracker[track_id]['looking_left']
                            right_count=sus_tracker[track_id]['looking_right']
                            sus_tracker[track_id]={"hand_below":0,"looking_left":0,"looking_right":0}
                            #print(sus_tracker)
                            if is_last_frame:
                                # For last frame, start from the last complete window
                                last_complete_window = (frame_count // 1800) * 1800
                                frame_threshold = last_complete_window
                            elif frame_count>1800:
                                frame_threshold = frame_count - 1800
                            else:
                                frame_threshold=0
                            
                            looking_left_flag = left_count >= 5
                            if (looking_left_flag):
                                cv2.putText(sharpened_image, f"Looking left {left_count} times", (80, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                                print(f"__________________looking left {left_count} times___________________")
                                tex=f"Looked left for {left_count} seconds"
                                send_suspicious_frame(track_img[track_id]["looking_left"],tex)

                            looking_right_flag = right_count >= 5
                            if(looking_right_flag):
                                cv2.putText(sharpened_image, f"Looking right {right_count} times", (80, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                                print(f"__________________looking right {right_count} times_________________")
                                tex=f"Looked right for {right_count} seconds"
                                send_suspicious_frame(track_img[track_id]["looking_right"],tex)

                            print("hand_below_duration = ",hand_below_duration)
                            hand_below_flag = hand_below_duration > 15
                            if (hand_below_flag):
                                    cv2.putText(sharpened_image, f"Hand below & looking down {hand_below_duration} sec", (80, 2000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                                    print(f"hand below and looking down more than {hand_below_duration} sec")
                                    tex=f"hand below more than {hand_below_duration} sec"

                                    send_suspicious_frame(track_img[track_id]["hand_below"],tex)
                    
                    mp_drawing.draw_landmarks(
                        sharpened_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    traceback.print_exc()
            snapshot_path = os.path.join(snapshot_dir, f"person_{track_id}.png")
            cv2.imwrite(snapshot_path, sharpened_image)
            del sharpened_image,copy
            gc.collect()    
        
        if is_last_frame:
            for track_id, values in sus_tracker.items():
                if any(values.values()):  # Check if any value is non-zero

                    track_id=track_id
                    hand_below_duration=values['hand_below']
                    left_count=values['looking_left']
                    right_count=values['looking_right']
                    if hand_below_duration>15:
                        tex=f"Hand below and looking down {hand_below_duration} seconds"
                        send_suspicious_frame(track_img[track_id]["hand_below"],tex)
                    if right_count>5:
                        tex=f"Looked right for {right_count} seconds"
                        send_suspicious_frame(track_img[track_id]["looking_right"],tex)
                    if left_count>5:
                        tex=f"Looked left for {left_count} seconds"
                        send_suspicious_frame(track_img[track_id]["looking_left"],tex)
            send_last_frame()
            print("..............GOING TO BREAK...................")
            break
        '''if is_last_frame:
            send_last_frame()'''
            # Save the processed image


def process_video_feed(video_path,venue_id):
    venue_id=venue_id
    global q2
    q2 = Queue()
    cap = cv2.VideoCapture(video_path)
    frame_interval = 30
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
            
        frame_count += 1
        if frame_count % frame_interval == 0:
            is_last_frame = frame_count + frame_interval >= total_frames
            results = model.track(frame, persist=True, classes=[0], conf=0.5)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            snapshot_dir = os.path.join(base_snapshot_dir, timestamp)
            os.makedirs(snapshot_dir, exist_ok=True)
            timestamp=datetime.now(timezone.utc)
            q2.put((frame, results, snapshot_dir, frame_count,timestamp,is_last_frame))
            del frame  # or any other large temporary object
            gc.collect()
        if stop_signal_received_exam==True:
            break 
            
    print("Finish")   
    
    cap.release()
    q2.put(None)