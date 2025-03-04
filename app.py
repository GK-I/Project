'''from flask import Flask, redirect,render_template
from flask import session,send_file
from flask import request
from flask import make_response
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin,LoginManager,login_user,logout_user,login_required,current_user,logout_user
from werkzeug.security import generate_password_hash,check_password_hash
from flask import flash
from datetime import datetime
import io

app=Flask(__name__)
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response
app.secret_key='MITS@123'

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///teachers.db'
db=SQLAlchemy(app)


login_manager=LoginManager()
login_manager.login_view='login'
login_manager.init_app(app)

class Teacher(UserMixin,db.Model):
    id=db.Column(db.Integer, primary_key=True,autoincrement=True)
    name=db.Column(db.String(100), nullable=False)
    email=db.Column(db.String(100), unique=True,nullable=False)
    password=db.Column(db.String(100),nullable=False)
    classes = db.relationship('Class', secondary='teacher_class', back_populates='teachers')

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    class_name = db.Column(db.String(100), nullable=False)
    
    # Relationship to Teachers
    teachers = db.relationship('Teacher', secondary='teacher_class', back_populates='classes')
    reports = db.relationship('Report', back_populates='classroom', cascade='all, delete-orphan')
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id', ondelete='CASCADE'), nullable=False)  # Foreign Key to Class
    pdf_file = db.Column(db.LargeBinary, nullable=False)  # To store the PDF file as binary data
    timestamp = db.Column(db.DateTime, default=datetime.now())  # Timestamp for when the report is created

    # Relationship with Class
    classroom = db.relationship('Class', back_populates='reports')  # Access class from Report
    


class TeacherClass(db.Model):
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id', ondelete='CASCADE'), primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id', ondelete='CASCADE'), primary_key=True)

app.app_context().push()
db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return Teacher.query.get(int(user_id))

@app.route('/')
def hello():
    return redirect(url_for('login'))

@app.route('/test-insert')
def test_insert():
    teacher = Teacher(name="Test Teacher", email="test@eduvision.com", password=generate_password_hash("password", method='pbkdf2:sha256'))
    db.session.add(teacher)
    class1 = Class(class_name="CS-A(S3)")
    class2 = Class(class_name="CS-B(S6)")
    class3 = Class(class_name="CY(S2)")
    class4 = Class(class_name="CS-A(S2)")
    
    # Add classes to session
    db.session.add_all([class1, class2, class3,class4])
    
    # Commit the teacher and class records to the database
    db.session.commit()
    
    # Now, associate the teacher with the classes
    teacher.classes.append(class1)
    teacher.classes.append(class2)
    teacher.classes.append(class3)
    teacher.classes.append(class4)
    db.session.commit()
    with open('D:/FinalProject/MediaPipeLearn/IJCSP23D1129.pdf', 'rb') as f:
        dummy_pdf_data = f.read()

    # Create 3 dummy reports for the test
    report1 = Report(classroom=class1, pdf_file=dummy_pdf_data)
    report2 = Report(classroom=class2, pdf_file=dummy_pdf_data)
    report3 = Report(classroom=class3, pdf_file=dummy_pdf_data)

    db.session.add_all([report1, report2, report3])
    db.session.commit()
    return "Inserted test record!"



@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        user=Teacher.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('profile'))
    return render_template("login.html")



@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'GET':
        return render_template("signup.html")
    
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    
    print(f"Received Name: {name}, Email: {email}")  # Debugging line to check if form data is correct
    
    user = Teacher.query.filter_by(email=email).first()
    if user:
        flash('Email already exists')
        return redirect(url_for('signup'))
    
    new_user = Teacher(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
    db.session.add(new_user)
    db.session.commit()
    
    print("User added to DB")  # Debugging line to confirm if user is added to the database
    flash('Account created successfully')
    return redirect(url_for('login'))



@app.route('/classroom_mode')
@login_required
def classroom_mode():
    # Assuming you pass the teacher's name as a parameter
    teacher_name = request.args.get('teacher_name')

    # Fetch the teacher object from the database
    teacher = Teacher.query.filter_by(name=teacher_name).first()
    print("Teacher Name from URL:", teacher_name)

    if teacher:
        # Pass teacher and associated classes to the template
        classes = teacher.classes  # Assuming teacher has a 'classes' relationship
        return render_template('classroom_mode.html', teacher=teacher, classes=classes)
    else:
        return "Teacher not found", 404
    

@app.route('/classroom/<int:class_id>/reports')
@login_required
def class_reports(class_id):
    # Fetch the class details and associated reports from the database
    class_details = Class.query.get(class_id)
    reports = Report.query.filter_by(class_id=class_id).all()
    
    return render_template('reports.html', class_details=class_details, reports=reports)


@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = Report.query.get(report_id)  # Retrieve the report by ID
    if report:
        pdf_content = report.pdf_file  # Get the binary content of the PDF file
        pdf_io = io.BytesIO(pdf_content)  # Convert binary data to a file-like object
        return send_file(pdf_io, as_attachment=True, download_name="report.pdf", mimetype='application/pdf')
    
    
@app.route('/profile')
@login_required
def profile():
    return render_template("home.html", user=current_user)

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    session.clear()
    logout_user()
    response = make_response(redirect(url_for("login")))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
    
app.run(debug=True)'''

from flask import Flask, redirect,render_template
from flask import session,send_file
import json
from flask import request
from flask import make_response
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin,LoginManager,login_user,logout_user,login_required,current_user,logout_user
from werkzeug.security import generate_password_hash,check_password_hash
from flask import flash,jsonify
from datetime import datetime
import base64
import io
import cv2
import os
import gc
from flask_cors import CORS
import numpy as np
from cv2 import dnn_superres
from ultralytics import YOLO
import mediapipe as mp
from fer import FER
import time
from datetime import datetime,timedelta
from queue import Queue
from threading import Thread
from flask_socketio import SocketIO
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from collections import defaultdict
import video_scanner
from exam_mode_implementation import process_video_feed, exam_processor,received_stop_signal,face
app=Flask(__name__)
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response
app.secret_key='MITS@123'
stop_signal_received = False
t1=None
t2=None
sk=None
isLastFrame=0
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///teachers.db'
app.config["VIDEO_BASE_DIR"] = "D:/FinalProject/MediaPipeLearn/Videos"
db=SQLAlchemy(app)
model = YOLO("yolo11n.pt")
socketio = SocketIO(app,max_http_buffer_size=100 * 1024 * 1024,cors_allowed_origins="*")

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=False)
# Initialize Super Resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
# Path to the FSRCNN model
path = "FSRCNN_x3.pb"
# Read the FSRCNN model
sr.readModel(path)
# Set the model type and x3scale
sr.setModel("fsrcnn", 3)
# Directory to save snapshots
base_snapshot_dir = "E:/classroom_mode_with_facemeshdemo2"
os.makedirs(base_snapshot_dir, exist_ok=True)
# Frame interval for snapshots
snapshot_interval_frames = 145
#start_time1 = time.time()
# Function to sharpen an image
#video_path = "demo1.mp4"
video_path="C:/Users/vinee/Downloads/demo2.mp4"
snapshot_queue = Queue()
embedder = FaceNet()

# Load face embeddings and trained model
data = np.load("C:/Users/vinee/Downloads/faces_embeddings_done_trial2.npz")
trainX, trainy = data['arr_0'], data['arr_1']
le = LabelEncoder()
trainy = le.fit_transform(trainy)
model1 = SVC(kernel='linear', probability=True)
model1.fit(trainX, trainy)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0,refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Initialize the face detector
detector = MTCNN()
student_reports_dict = defaultdict(lambda: {"instances": 0, "attentive": 0, "moderately_attentive": 0, "inattentive": 0})
def get_embedding(face_img):
    """Extracts FaceNet embedding from an image."""
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)  
    return yhat[0]

def send_suspicious_frame(image,text):
    image=cv2.resize(image,(800,800))
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Convert image to Base64
    
    # Send frame using Flask-SocketIO
    socketio.emit('suspicious_frame', {'image': encoded_image,'reason':text})
def send_last_frame():
    socketio.emit('Last frame',{})

def process_image(t_im):
    """Processes an image: detects largest face, extracts embedding, and labels it."""
    try:
        if t_im is None:
            print(f"Error reading image")
            return

        t_im = cv2.cvtColor(t_im, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(t_im)

        if not faces:
            print(f"No face found ")
            return  # Skip processing if no face detected

        # Select the largest face
        largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
        x, y, w, h = largest_face['box']

        # Extract and preprocess the face
        face_img = t_im[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (160, 160))
        test_im = get_embedding(face_img)
        test_im = [test_im]

        # Predict label and confidence score
        ypreds = model1.predict(test_im)
        yprobs = model1.predict_proba(test_im).max()

        # Set a threshold for unknown faces (adjustable)
        THRESHOLD = 0.5  
        if yprobs < THRESHOLD:
            name = "Unknown"
        else:
            name = le.inverse_transform(ypreds)[0]

        # Draw bounding box and label
        cv2.rectangle(t_im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(t_im, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed image
        t_im = cv2.cvtColor(t_im, cv2.COLOR_RGB2BGR)
        del t_im,largest_face,test_im
        print(f"Processed and saved:  - {name}")
        return (name,face_img)
    except Exception as e:
        print(f"Error processing : {e}")


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Compute vectors
    ba = a - b
    bc = c - b
    # Calculate angle using dot product and arccos
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Clip to handle precision issues
    return angle

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def snapshot_processor():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils
    stop=0

    while True:
        item = snapshot_queue.get()
        if item is None:
            break
        people,attentive, inattentive, moderate, happy, sad, fear, disgust, neutral, angry,surprise,handraisecount,slouchedcount,handsonheadcount,extremeendcount,normalcount,handsbelowcount = [0] * 17
        if stop:
            break
        frame, results, snapshot_dir,timestamp,isLastFrame= item
        if isLastFrame:
            
            print("Last Frame going to be sent")
            while (not snapshot_queue.empty()):
                snapshot_queue.get()
                if snapshot_queue.empty():
                    stop=1
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            cropped_person = frame[y1:y2, x1:x2]
            people+=1

            # Apply super resolution and sharpening
            result = sr.upsample(cropped_person)
            sharpened_image = sharpen_image(result)

            # Save the cropped individual image
            snapshot_path = os.path.join(snapshot_dir, f"person_{i+1}.png")
            #cv2.imwrite(snapshot_path, sharpened_image)
                # Perform emotion detection
            dominant_emotion, emotion_score = emotion_detector.top_emotion(sharpened_image)
            emotion_text = (
                    f"Emotion: {dominant_emotion} (Score: {emotion_score:.2f})"
                    if dominant_emotion else "No emotions detected"
            )
            hand_raise=0
            hand_below=0
            extreme_end=0
            hand_on_face=0
            slouched=0
            normal=0
            score=0
            # Perform pose detection
            image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            (name,face_img)=process_image(sharpened_image)
            print("CURRENTLY PROCESSING="+name)
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
                    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = range(6)

                    '''shoulderl = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x*image_width ,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*image_height ]
                        elbowl = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x*image_width ,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y*image_height ]
                        wristl = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x*image_width ,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y *image_height]
                        shoulderr=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*image_width,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*image_height]
                        elbowr = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x*image_width ,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y*image_height ]
                        wristr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x *image_width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y*image_height]'''
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
                    left_pinky=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_PINKY]
                    right_pinky=pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_PINKY]
                    #Raising hands
                    wristl = landmarks_array[2]  # Left wrist
                    shoulderl = landmarks_array[0]  # Left shoulder
                    leftangle = calculate_angle(landmarks_array[0], landmarks_array[1], landmarks_array[2])  # Left angle (shoulder -> elbow -> wrist)

                        # Right wrist, shoulder, and angles for hand raising condition
                    wristr = landmarks_array[5]  # Right wrist
                    shoulderr = landmarks_array[3]  # Right shoulder
                    elbowl=landmarks_array[1]
                    elbowr=landmarks_array[4]
                    rightangle = calculate_angle(landmarks_array[3], landmarks_array[4], landmarks_array[5])  # Right angle (shoulder -> elbow -> wrist)
                    print(leftangle)
                    print(emotion_text)
                    print(rightangle)
                    print(timestamp)
                    if (wristl[1] < shoulderl[1] and leftangle > 45) or (wristr[1] < shoulderr[1] and rightangle > 45):  # Angle close to 180 for a raised hand
                        print("RAISING HANDS")
                        hand_raise=1
                        handraisecount+=1
                    leftangle = calculate_angle(shoulderl, elbowl, wristl)
                    rightangle = calculate_angle(shoulderr, elbowr, wristr)
                    if ((wristl[1] > elbowl[1] and leftangle >= 160 and (not(left_wrist.visibility>0.80))) or 
                        (wristr[1] > elbowr[1] and rightangle >= 160 and (not(right_wrist.visibility>0.80)))):
                        print("hand below"+timestamp)  
                        hand_below = 1
                        handsbelowcount+=1
                    else:
                        hand_below = 0
                    face_rgb=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) 
                    (x,y,p1,p2,face_landmarks,face_landmarks_detected)=face(image_rgb,sharpened_image)
                    if face_landmarks_detected==True:
                            
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
                    #Extreme left/right
                    half=(left_shoulder.x+right_shoulder.x)/2
                    l=(left_eye_outer.x+left_ear.x)/ 2
                    r=(right_eye_outer.x+right_ear.x)/ 2

                    if face_landmarks_detected==True:
                        if (y<-5 and x>-5.5):
                            extreme_end=1
                            extremeendcount+=1
                            cv2.putText(sharpened_image, "Looking right", (80, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        elif ((y>5 and x>-5.5)):
                            extreme_end=1
                            extremeendcount+=1
                            cv2.putText(sharpened_image, "Looking left", (80, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    elif(face_landmarks_detected==False and l < half):
                        extreme_end=1
                        extremeendcount+=1
                        cv2.putText(sharpened_image, "Looking right", (80, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    elif(face_landmarks_detected==False and r > half):
                        extreme_end=1
                        extremeendcount+=1
                        cv2.putText(sharpened_image, "Looking left", (80, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    '''half=(left_shoulder.x+right_shoulder.x)/2
                    #tolerance = 0.05  # Adjust as needed based on coordinate range
                    if ((left_eye_outer.x + left_ear.x) / 2) < half  or ((right_eye_outer.x + right_ear.x) / 2) > half :
                        extreme_end = 1
                        extremeendcount+=1
                    else:
                        extreme_end = 0
                    #Hands on face/head'''

                    # Calculate the bounding box for the head region
                    min_x = min(nose.x, left_eye.x, right_eye.x, left_ear.x, right_ear.x)
                    max_x = max(nose.x, left_eye.x, right_eye.x, left_ear.x, right_ear.x)
                    min_y = min(nose.y, left_eye.y, right_eye.y, left_ear.y, right_ear.y)
                    max_y = max(nose.y, left_eye.y, right_eye.y, left_ear.y, right_ear.y)

                    # Expand the box slightly to include the jaw area
                    box_margin_x = 0.1  # Adjust margin as needed
                    box_margin_y= 0.1
                    min_x = max(0, min_x - box_margin_x)  # Ensure within bounds
                    max_x = min(1, max_x + box_margin_x)
                    min_y = max(0, min_y - box_margin_y)
                    max_y = min(1, max_y + box_margin_y)

                        # Convert normalized coordinates to pixel coordinates
                        #min_x_px = int(min_x * image_width)
                        #max_x_px = int(max_x * image_width)
                        #min_y_px = int(min_y * image_height)
                        #max_y_px = int(max_y * image_height)

                        # Draw the ROI on the image
                    roi_color = (0, 255, 0)  # Green for ROI box
                    #cv2.rectangle(sharpened_image, (min_x_px, min_y_px), (max_x_px, max_y_px), roi_color, 2)
                    # Check if either wrist is inside the bounding box
                    left_hand_in_box = (min_x <=(left_wrist.x+left_pinky.x)/2 <= max_x) and (min_y <= (left_wrist.y+left_pinky.y)/2 <= max_y)
                    right_hand_in_box = (min_x <= (right_wrist.x+right_pinky.x)/2 <= max_x) and (min_y <= (right_wrist.y+right_pinky.y)/2 <= max_y)

                        
                    if left_hand_in_box or right_hand_in_box:
                        hand_on_face=1
                        handsonheadcount+=1
                        #cv2.putText(annotated_image, "Hand near head!", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    #shoulderdiff(not good posture)
                    # Calculate the absolute difference between the Y-coordinates
                    shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
                    if shoulder_height_diff>=0.07:
                        slouched=1
                        slouchedcount+=1
                        #cv2.putText(annotated_image, "Slouched posture", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

            except:
                pass



                #now using pose landmarks to gain insights about posture, add to sqlalchemy db
            match dominant_emotion:
                case "happy":
                    happy+=1
                    score+=2
                case "sad":
                    sad+=1
                    score-=2
                case "neutral":
                    neutral+=1
                    score+=1
                case "angry":
                    angry+=1
                    score-=1
                case "disgust":
                    disgust+=1
                    score-=2
                case "fear":
                    fear+=1
                    score-=2
                case "surprise":
                    surprise+=1
                    score+=2
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                        sharpened_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
            if extreme_end==0 and slouched==0 and hand_on_face==0 and hand_below==0:
                normal=1
                normalcount+=1
                score+=2
            else:
                normal=0
            if hand_raise==1:
                slouched=0
                extreme_end=0
                hand_on_face=0
                hand_below=0
                cv2.putText(
                    sharpened_image,name+"-"+emotion_text+",Raising hands", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
                score+=5
            if extreme_end==1:
                cv2.putText(
                        sharpened_image,name+"-"+emotion_text+",extreme looking", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                    )
                score-=3

            if normal==1:
                cv2.putText(
                    sharpened_image,name+"-"+emotion_text+",Normal posture", (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
            if hand_on_face==1:
                cv2.putText(
                    sharpened_image,name+"-"+emotion_text+",Hands on face", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
                score-=3
            if hand_below==1:
                cv2.putText(
                    sharpened_image,name+"-"+emotion_text+",Hands below", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
                score-=2
                    
            if slouched==1:
                cv2.putText(
                    sharpened_image,name+"-"+emotion_text+",Slouched posture", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
                score-=3
                
            if score>=3:
                cv2.putText(
                    sharpened_image, "ATTENTIVE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                    )
                status="ATTENTIVE"
                attentive+=1
            elif (score>=0 and score<=2)or score==-1:
                cv2.putText(sharpened_image,"MODERATELY ATTENTIVE",(10,120),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 3)
                print("hi")
                moderate+=1
                status="MODERATELY ATTENTIVE"
            else:
                cv2.putText(sharpened_image,"INATTENTIVE",(10,120),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 3)
                print("hello")
                inattentive+=1
                status="INATTENTIVE"

            if name in student_reports_dict:
                student_reports_dict[name]["instances"]+=1
                if status=="ATTENTIVE":
                    student_reports_dict[name]["attentive"]+=1
                elif status=="MODERATELY ATTENTIVE":
                    student_reports_dict[name]["moderately_attentive"]+=1
                else:
                    student_reports_dict[name]["inattentive"]+=1
            else:
                    if name!="Unknown":
                        student_reports_dict[name] = {
        "instances": 1,
        "attentive": 1 if status == "ATTENTIVE" else 0,
        "moderately_attentive": 1 if status == "MODERATELY ATTENTIVE" else 0,
        "inattentive": 1 if status == "INATTENTIVE" else 0,
    }

                

            #Save annotated image with pose and emotion
            annotated_path = os.path.join(snapshot_dir, f"annotated_person_{i+1}.png")
            cv2.imwrite(annotated_path, sharpened_image)
            del sharpened_image,cropped_person,result
            gc.collect()

        snapshot_queue.task_done()
        attentivepercentage=(attentive/people)*100
        inattentivepercentage=(inattentive/people)*100
        moderatepercentage=(moderate/people)*100
        happypercentage=(happy/people)*100
        sadpercentage=(sad/people)*100
        fearpercentage=(fear/people)*100
        disgustpercentage=(disgust/people)*100
        neutralpercentage=(neutral/people)*100
        angrypercentage=(angry/people)*100
        surprisepercentage=(surprise/people)*100
        
        attentiveness_data = {
    "attentive": attentivepercentage,
    "inattentive": inattentivepercentage,
    "moderately_attentive": moderatepercentage,  
    "timestamp": timestamp,  # Ensure timestamp is ISO 8601 or Unix format
    "box_coordinate": [x1, y1, x2, y2],
    "emotions": [
        happypercentage, sadpercentage, fearpercentage,
        angrypercentage, disgustpercentage, surprisepercentage, neutralpercentage
    ],"posture": [  normalcount,handraisecount,handsbelowcount,extremeendcount,handsonheadcount,slouchedcount        
    ],"isLastFrame":isLastFrame 
        }

        # Emit the data to the frontend
        socketio.emit('attentiveness_update', attentiveness_data)

    pose.close()

def processvideo():
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    print("...................video started................")
    #worker_thread = Thread(target=snapshot_processor, args=(snapshot_queue,))
    #worker_thread.start()
    global stop_signal_received
    global isLastFrame
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("breakingg")
            break
        '''_, buffer = cv2.imencode('.jpg', frame)
        
        # Convert the frame to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit the base64 frame to the frontend via SocketIO
        socketio.emit('frame_update', {'image': jpg_as_text})'''

        frame_count += 1
        # Process every 100th frame
        if frame_count % snapshot_interval_frames == 0:
            results = model.predict(frame, classes=[0],conf=0.5)  # Detect people in the frame
            
            # Create a timestamp folder for the frame
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = datetime.now().strftime("%H_%M_%S")
            snapshot_dir = os.path.join(base_snapshot_dir, timestamp)
            os.makedirs(snapshot_dir, exist_ok=True)
            if stop_signal_received:
                isLastFrame=1
                print("islastframe forwarded")
            # Queue the frame for snapshot processing
            snapshot_queue.put((frame, results, snapshot_dir,timestamp,isLastFrame))
            if stop_signal_received:
                break
            del frame
            gc.collect()
    snapshot_queue.put(None)
        # End timer for FPS calculation
        

    # Cleanup
    cap.release()
    #cv2.destroyAllWindows()
    #snapshot_queue.put(None)
    #worker_thread.join()
    
    end_time1 = time.time()
    #print(end_time1-start_time1)


login_manager=LoginManager()
login_manager.login_view='login'
login_manager.init_app(app)

class Teacher(UserMixin,db.Model):
    id=db.Column(db.Integer, primary_key=True,autoincrement=True)
    name=db.Column(db.String(100), nullable=False)
    email=db.Column(db.String(100), unique=True,nullable=False)
    password=db.Column(db.String(100),nullable=False)
    classes = db.relationship('Class', secondary='teacher_class', back_populates='teachers')
    reports = db.relationship('Report', back_populates='teacher')
    student_reports = db.relationship('StudentReport', back_populates='teacher', cascade='all, delete-orphan')

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    class_name = db.Column(db.String(100), nullable=False)
    
    # Relationship to Teachers
    teachers = db.relationship('Teacher', secondary='teacher_class', back_populates='classes')
    reports = db.relationship('Report', back_populates='classroom', cascade='all, delete-orphan')
    student_reports = db.relationship('StudentReport', back_populates='classroom', cascade='all, delete-orphan')

class ExamVenue(db.Model):
    venue_id = db.Column(db.Integer, primary_key=True)  
    venue_name = db.Column(db.String(100), nullable=False)  

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id', ondelete='CASCADE'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id', ondelete='CASCADE'), nullable=False)
    pdf_file = db.Column(db.LargeBinary, nullable=True)  
    json_data = db.Column(db.Text, nullable=True)  # To store JSON as text
    timestamp = db.Column(db.DateTime, default=datetime.now()) 
    # Relationship with Class
    classroom = db.relationship('Class', back_populates='reports')  # Access class from Report
    teacher = db.relationship('Teacher', back_populates='reports')
    student_reports = db.relationship('StudentReport', back_populates='report', cascade='all, delete-orphan')

class StudentReport(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)  # Student Name
    report_id = db.Column(db.Integer, db.ForeignKey('report.id', ondelete='CASCADE'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id', ondelete='CASCADE'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id', ondelete='CASCADE'), nullable=False)
    
    # Attentiveness tracking
    attentive = db.Column(db.Integer, default=0, nullable=False)  # Count of attentive instances
    inattentive = db.Column(db.Integer, default=0, nullable=False)  # Count of inattentive instances
    moderately_attentive = db.Column(db.Integer, default=0, nullable=False)  # Count of moderate attentiveness
    instances = db.Column(db.Integer, default=0, nullable=False)  # Total times student was detected

    # Relationships
    report = db.relationship('Report', back_populates='student_reports')
    classroom = db.relationship('Class', back_populates='student_reports')
    teacher = db.relationship('Teacher', back_populates='student_reports')

class TeacherClass(db.Model):
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id', ondelete='CASCADE'), primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id', ondelete='CASCADE'), primary_key=True)

class ExamReport(db.Model):
    report_id = db.Column(db.Integer, primary_key=True)
    venue_id = db.Column(db.Integer, db.ForeignKey('exam_venue.venue_id'), nullable=False)
    report_name = db.Column(db.String(200), nullable=False)
    pdf_file = db.Column(db.LargeBinary, nullable=True)

class ExamVideo(db.Model):
    video_id = db.Column(db.Integer, primary_key=True)
    venue_id = db.Column(db.Integer, db.ForeignKey('exam_venue.venue_id'), nullable=False)
    video_name = db.Column(db.String(200), nullable=False)
    video_path = db.Column(db.String(300), nullable=False)  # Path to video file
    processed = db.Column(db.Boolean, default=False)

app.app_context().push()
db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return Teacher.query.get(int(user_id))

@app.route('/')
def hello():
    return redirect(url_for('login'))

@app.route('/test-insert')
def test_insert():
    teacher = Teacher(name="Test Teacher", email="test@eduvision.com", password=generate_password_hash("password", method='pbkdf2:sha256'))
    db.session.add(teacher)
    class1 = Class(class_name="CS-A(S3)")
    class2 = Class(class_name="CS-B(S6)")
    class3 = Class(class_name="CY(S2)")
    class4 = Class(class_name="CS-A(S2)")
    
    # Add classes to session
    db.session.add_all([class1, class2, class3,class4])
    
    # Commit the teacher and class records to the database
    db.session.commit()
    
    # Now, associate the teacher with the classes
    teacher.classes.append(class1)
    teacher.classes.append(class2)
    teacher.classes.append(class3)
    teacher.classes.append(class4)
    db.session.commit()
    with open('D:/FinalProject/MediaPipeLearn/IJCSP23D1129.pdf', 'rb') as f:
        dummy_pdf_data = f.read()

    # Create 3 dummy reports for the test
    report1 = Report(classroom=class1, teacher=teacher, pdf_file=dummy_pdf_data, json_data='{}')
    report2 = Report(classroom=class2, teacher=teacher, pdf_file=dummy_pdf_data, json_data='{}')
    report3 = Report(classroom=class3, teacher=teacher, pdf_file=dummy_pdf_data, json_data='{}')

    db.session.add_all([report1, report2, report3])
    db.session.commit()
    return "Inserted test record!"

@app.route('/test-insert2')
def test_insert2():
    teacher = Teacher(name="Goutham", email="sgouthamkrishna123@gmail.com", password=generate_password_hash("password", method='pbkdf2:sha256'))
    db.session.add(teacher)
    class1 = Class(class_name="CS-A(S3)")
    class2 = Class(class_name="CS-B(S6)")
    class3 = Class(class_name="CY(S2)")
    class4 = Class(class_name="CS-A(S2)")
    
    # Add classes to session
    db.session.add_all([class1, class2, class3,class4])
    
    # Commit the teacher and class records to the database
    db.session.commit()
    
    # Now, associate the teacher with the classes
    teacher.classes.append(class1)
    teacher.classes.append(class2)
    teacher.classes.append(class3)
    teacher.classes.append(class4)
    db.session.commit()


    return "Inserted test record!"



@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        user=Teacher.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('profile'))
    return render_template("login.html")



@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'GET':
        return render_template("signup.html")
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    print(f"Received Name: {name}, Email: {email}")  # Debugging line to check if form data is correct
    user = Teacher.query.filter_by(email=email).first()
    if user:
        flash('Email already exists')
        return redirect(url_for('signup'))
    new_user = Teacher(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
    db.session.add(new_user)
    db.session.commit()
    print("User added to DB")  # Debugging line to confirm if user is added to the database
    flash('Account created successfully')
    return redirect(url_for('login'))
session_context = {}
@app.route('/start_session')
@login_required
def start_session():
    global stop_signal_received
    global isLastFrame
    global student_reports_dict
    student_reports_dict = defaultdict(lambda: {"instances": 0, "attentive": 0, "moderately_attentive": 0, "inattentive": 0})
    teacher_id=request.args.get('teacher_id')
    class_id=request.args.get('class_id')
    print(teacher_id)
    print(class_id)
    t1=Thread(target=processvideo, daemon=True).start()
    # Start the snapshot processor worker thread
    time.sleep(0.5)
    t2=Thread(target=snapshot_processor, daemon=True).start()
    stop_signal_received = False
    isLastFrame=0
    return render_template("session.html", user=current_user)
def send_suspicious_frame(image,text):
    image=cv2.resize(image,(800,800))
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Convert image to Base64
    
    # Send frame using Flask-SocketIO
    socketio.emit('suspicious_frame', {'image': encoded_image,'reason':text})
@app.route('/start_exam',methods=['GET','POST'])
@login_required
def start_exam():
    global stop_signal_received
    global isLastFrame
    venue_id = request.form.get('venue_id')  # Get venue_id from the form
    video_id = request.form.get('video_id')  # Get video_id from the form

    if not venue_id or not video_id:
        flash("Missing venue_id or video_id!", "error")
        return redirect(url_for('venue_dashboard', venue_id=venue_id))

    print(f"Received Video ID: {video_id}, Venue ID: {venue_id}")  # Debugging

    # Fetch the video path
    video = ExamVideo.query.get(video_id)  # Query the database for the video
    if not video:
        flash("Selected video not found.", "error")
        return redirect(url_for('venue_dashboard', venue_id=venue_id))

    video_path = video.video_path.replace('\\', '/')  # Extract video path
    #global student_reports_dict
    #student_reports_dict = defaultdict(lambda: {"instances": 0, "attentive": 0, "moderately_attentive": 0, "inattentive": 0})

    t1=Thread(target=process_video_feed,args=(video_path,venue_id) ,daemon=True).start()
    # Start the snapshot processor worker thread
    t2=Thread(target=exam_processor, daemon=True).start()
    stop_signal_received = False
    isLastFrame=0
    return render_template("exam_session.html", venue_id=venue_id,video_id=video_id)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    print("🔍 Request received!")  # Debugging Log

    # Check if required fields exist
    if "pdf" not in request.files:
        print("❌ Missing PDF file")
        return jsonify({"error": "Missing PDF file"}), 400
    if "venue_id" not in request.form:
        print("❌ Missing venue_id")
        return jsonify({"error": "Missing venue_id"}), 400
    if "video_id" not in request.form:
        print("❌ Missing video_id")
        return jsonify({"error": "Missing video_id"}), 400

    pdf_file = request.files["pdf"]
    pdf_data = pdf_file.read()
    venue_id = request.form["venue_id"]
    video_id = request.form["video_id"]

    # Check if venue_id and video_id are integers
    try:
        venue_id = int(venue_id)
        video_id = int(video_id)
    except ValueError:
        print("❌ Invalid venue_id or video_id")
        return jsonify({"error": "Invalid venue_id or video_id"}), 400

    # Fetch the video details
    video = ExamVideo.query.get(video_id)
    if not video:
        print("❌ Video not found")
        return jsonify({"error": "Video not found"}), 404

    #video_filename = video.video_path.split("/")[-1]
    video_filename=video.video_name.split('.')[-2]
    report_filename = f"{video_filename}.pdf"

    # Save to database
    new_report = ExamReport(
        venue_id=venue_id,
        report_name=report_filename,
        pdf_file=pdf_data
    )

    db.session.add(new_report)
    db.session.commit()

    print("✅ PDF successfully saved!")
    return jsonify({"message": "PDF saved successfully", "report_id": new_report.report_id}), 200


@socketio.on('save_session_data')
def handle_session_data(session_data):
    # Existing code...
    
    # Add PDF handling if pdfData exists
    if session_data.get('pdfData'):
        pdf_data = base64.b64decode(session_data['pdfData'].split(',')[1])
        
        new_report = Report(
            class_id=session_data['class_id'],
            teacher_id=session_data['teacher_id'],
            json_data=json.dumps(session_data),
            pdf_file=pdf_data
        )
        db.session.add(new_report)
        db.session.commit()

# Handle receiving session data from the client
@socketio.on('sessionData')
def handle_session_data(sessionData):
    try:
        #print("Received session data:", sessionData)

        # Extract required fields from sessionData
        class_id = session_context['class_id']
        teacher_id = session_context['teacher_id']
        pdf_data = sessionData.get('pdfData')  # This is expected to be in base64 format
        json_data = json.dumps(sessionData)  # Convert entire sessionData to a JSON string

        # Decode PDF data if available
        pdf_binary = None
        if pdf_data:
            try:
                pdf_binary = base64.b64decode(pdf_data.split(',')[1])  # Remove "data:application/pdf;base64," prefix
            except Exception as e:
                print(f"Error decoding PDF data: {e}")

        # Create a new Report instance
        report = Report(
            class_id=class_id,
            teacher_id=teacher_id,
            pdf_file=pdf_binary,  # Store binary PDF data if available
            json_data=json_data
        )

        # Save the report to the database
        db.session.add(report)
        db.session.commit()
        print("Report saved successfully.")
        bulk_insert_data = [
            StudentReport(
                name=student_name,
                class_id=class_id,
                teacher_id=teacher_id,
                report_id=report.id,
                instances=data["instances"],
                attentive=data["attentive"],
                moderately_attentive=data["moderately_attentive"],
                inattentive=data["inattentive"]
            )
            for student_name, data in student_reports_dict.items()
        ]

        db.session.bulk_save_objects(bulk_insert_data)
        db.session.commit() 
        print("___________ALL STUDENTS ALSO UPDATED____________")
    except Exception as e:
        db.session.rollback()
        print(f"Error saving session data: {e}")
@app.route("/report/<int:report_id>")
def show_report(report_id):
    
    teacher_id=session_context['teacher_id'] 
    class_id=session_context['class_id']
    student_reports = StudentReport.query.filter_by(
        teacher_id=teacher_id, class_id=class_id, report_id=report_id
    ).all()


    students_data = [
        {
            "name": student.name,
            "attentive": int((student.attentive/student.instances)*100),
            "moderately_attentive": int((student.moderately_attentive/student.instances)*100),
            "inattentive": int((student.inattentive/student.instances)*100)
        }
        for student in student_reports
    ]

    return render_template("student_report.html", students=students_data)

@socketio.on('stop_session')
def handle_stop_session(message=None):
    global stop_signal_received,t1,t2
    stop_signal_received = True
    received_stop_signal(stop_signal_received)
    print("Stop signal received")
    # Ensure threads stop gracefully
    if t1 and t1.is_alive():
        t1.join()
    if t2 and t2.is_alive():
        t2.join()
    


'''@app.route('/classroom_mode')
@login_required
def classroom_mode():
    # Assuming you pass the teacher's name as a parameter
    teacher_name = request.args.get('teacher_name')

    # Fetch the teacher object from the database
    teacher = Teacher.query.filter_by(name=teacher_name).first()
    print("Teacher Name from URL:", teacher_name)

    if teacher:
        # Pass teacher and associated classes to the template
        classes = teacher.classes  # Assuming teacher has a 'classes' relationship
        return render_template('classroom_mode.html', teacher=teacher, classes=classes)
    else:
        return "Teacher not found", 404'''
@app.route('/classroom_mode')
@login_required
def classroom_mode():
    teacher_name = request.args.get('teacher_name')
    teacher = Teacher.query.filter_by(name=teacher_name).first()
    teacher_id=Teacher.query.filter_by(name=teacher_name).first().id
    session_context['teacher_id']=teacher_id

    if teacher:
        classes = teacher.classes

        # Delete reports older than 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        reports_to_delete = Report.query.filter(Report.timestamp < seven_days_ago).all()

        if reports_to_delete:  # Check if any reports need to be deleted
            try:
                for report in reports_to_delete:
                    db.session.delete(report)
                db.session.commit()
                print(f"Deleted {len(reports_to_delete)} reports older than 7 days.")
            except Exception as e:
                db.session.rollback()  # Rollback on error
                print(f"Error deleting reports: {e}")
                # Consider logging the error or displaying a message to the user

        return render_template('classroom_mode.html', teacher=teacher, classes=classes)
    else:
        return "Teacher not found", 404
    

@app.route('/exam_mode',methods=['GET','POST'])
@login_required
def exam_mode():
    # Fetch all the venues from the database
    teacher_name = request.args.get('teacher_name')
    teacher = Teacher.query.filter_by(name=teacher_name).first()
    venues = ExamVenue.query.all()
    video_scanner.scan_and_update_videos(app)
    return render_template('exam_mode.html', venues=venues,teacher=teacher)

@app.route('/venue/<int:venue_id>',methods=['GET','POST'])
def venue_dashboard(venue_id):
    venue = ExamVenue.query.get_or_404(venue_id)
    reports = ExamReport.query.filter_by(venue_id=venue_id).all()
    videos = ExamVideo.query.filter_by(venue_id=venue_id, processed=False).all()  # Show only unprocessed videos

    return render_template('venue_dashboard.html', 
                           venue_name=venue.venue_name, 
                           venue_id=venue_id, 
                           reports=reports, 
                           videos=videos)

@app.route('/classroom/<int:class_id>/reports')
@login_required
def class_reports(class_id):
    # Fetch the class details and associated reports from the database
    class_details = Class.query.get(class_id)
    print(class_id)
    session_context['class_id']=class_id
    #reports = Report.query.filter_by(class_id=class_id).all()
    report = Report.query.filter_by(class_id=class_id,teacher_id=current_user.id).all()
    return render_template('reports.html',reports=report,class_id=class_id,teacher_id=current_user.id)


@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = Report.query.get(report_id)  # Retrieve the report by ID
    if report:
        pdf_content = report.pdf_file  # Get the binary content of the PDF file
        pdf_io = io.BytesIO(pdf_content)  # Convert binary data to a file-like object
        return send_file(pdf_io, as_attachment=True, download_name="report.pdf", mimetype='application/pdf')
@app.route('/download_exam_report/<int:report_id>')
@login_required
def download_exam_report(report_id):
    report = ExamReport.query.get(report_id)  # Retrieve the report by ID
    if report:
        pdf_content = report.pdf_file  # Get the binary content of the PDF file
        pdf_io = io.BytesIO(pdf_content)  # Convert binary data to a file-like object
        return send_file(pdf_io, as_attachment=True, download_name="report.pdf", mimetype='application/pdf')

@app.route('/view_report/<int:report_id>')
@login_required  # If you have login required
def view_report(report_id):
    return render_template('view_report.html', report_id=report_id)  # Pass report_id to template

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('get_report_data')  # New event for client request
def handle_get_report_data(report_id):
    print(f"Client requested report data for ID: {report_id}")
    try:
        report = Report.query.get(report_id)
        session_data = json.loads(report.json_data)
        if 'pdfData' in session_data:
            del session_data['pdfData']
        socketio.emit('report_data', session_data, to=request.sid) # Send only to the client who requested it
        print("Report data sent successfully")
    except Exception as e:
        print(f"Error getting or sending report data: {e}")
        socketio.emit('report_error', {'message': 'Error retrieving report data'}, to=request.sid) # Send error message to the client


    
@app.route('/profile')
@login_required
def profile():
    return render_template("home.html", user=current_user)

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    session.clear()
    logout_user()
    response = make_response(redirect(url_for("login")))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
    
#app.run(debug=True,threaded=True)
if __name__ == "__main__":
    from app import app
    socketio.run(app, debug=False)