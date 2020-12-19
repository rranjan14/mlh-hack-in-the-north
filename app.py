from flask import Flask, render_template, Response, request, redirect,url_for,session
import json
import sys
import argparse
import cv2
from time import time
import os
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.liveness    import FaceLivenessModels, FaceLiveness
FAKE=False
FACE_ID="NOT APPLICABLE"
UPLOAD_FOLDER='/datasets'
 
# Use flask for web app
app = Flask(__name__)
 
 
# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS        = "models/liveness/"
 
# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)
 
 
 
def cam_init(cam_index, width, height): 
    cap = cv2.VideoCapture(cam_index)
    if sys.version_info < (3, 0):
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
 
 
def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        if confidence is not None:
            text = "{} {:.2f}%".format(face_id, confidence)
        else:
            text = "{}".format(face_id)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
 
def process_facerecognition():
    
    cam_index = 0
    cam_resolution = RESOLUTION_QVGA
    model_detector=FaceDetectorModels.HAARCASCADE
    model_recognizer=FaceEncoderModels.LBPH
 
 
 
    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])
 
    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
        # Initialize face recognizer
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        face_liveness  = FaceLiveness(model=FaceLivenessModels.EYESBLINK_MOUTHOPEN, path=INPUT_DIR_MODEL_LIVENESS)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)
 
 
    except: 
        face_encoder = None
        print("Warning, check if models and trained dataset models exists!")
 
    face_id, confidence = (None, 0)
    eyes_close, eyes_ratio = (False, 0)
    total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 1)
    mouth_open, mouth_ratio = (False, 0)
    total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 1)
 
    time_start = time()
    time_elapsed = 0
    runtime = 30
    
    print("Note: this will run for {} seconds only".format(runtime))
    while (True):
 
        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break
 
 
        # Detect and identify faces in the frame
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
            mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)
            is_fake_print  = face_liveness2.is_fake(frame, face)
            FAKE=is_fake_print
            (x, y, w, h) = face
 
            if is_fake_print:
                face_id, confidence = ("Fake", None)
            else:
 
                # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
                if face_encoder is not None:
                    face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
            # Set text and bounding box on face
            label_face(frame, (x, y, w, h), face_id, confidence)
 
            # Process 1 face only
            break
 
        time_elapsed = time()-time_start
        if time_elapsed>runtime:
            if not FAKE:
                FACE_ID=face_id
                print(FAKE)
                print(FACE_ID)
 
        # Display updated frame to web app
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
 
    # Release the camera
    camera.release()
    cv2.destroyAllWindows()
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
PREV_URL='register'
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def index():
    return render_template('home.html')
 
 
@app.route('/login')
def login():
    PREV_URL='login'
    return render_template('login.html')
 
@app.route('/login-user', methods=['POST'])
def login_user():
    if request.method == 'POST':
        FACE_ID = request.form["Name"]
        return redirect('/detection')
 
 
@app.route('/register')
def register():
    PREV_URL='register'
    return render_template('register.html')
 
 
@app.route('/register-user', methods=['POST'])
def register_user():
    if request.method == 'POST':
        name = request.form["Name"]
        name = name.replace(" ", "").strip()
        name = name.lower()
        print(name)
        aadhar = request.form["Aadhar"]
        aadhar = aadhar.replace("-", "").strip()
        dir_name = name+"-"+aadhar
        path = os.path.join(UPLOAD_FOLDER, dir_name)
        if not os.path.exists(path):
            os.mkdir()
        file = request.files['Photo']
        if file and allowed_file(file.filename):
            if not os.path.exists(os.path.join(path, file.filename)):
                file.save(os.path.join(path, file.filename))
        return redirect('/detection')
 
@app.route('/detection')
def detection():
    if(PREV_URL=='register'):
        os.system('python3 training.py --classifier 0 --encoder 0 --classifier 0')
    return render_template('detection.html')
 
 
# Entry point for web app
 
 
@app.route('/video_viewer')
def video_viewer():
    return Response(process_facerecognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route("/certificate")
def certificate():
    return render_template('certificate.html',FAKE=FAKE,FACE_ID=FACE_ID)
 
 
if __name__=="__main__":
    print("\n\nNote: Open browser and type http://127.0.0.1:5000/ or http://ip_address:5000/ \n\n")
    # Run flask for web app
    app.run(host='0.0.0.0', threaded=True, debug=True)
 

