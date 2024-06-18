from flask import Flask, render_template, Response
import cv2
import yaml
import numpy as np

app = Flask(__name__)

# Load parking spot coordinates from YAML file
fn_yaml = r"./datasets/video.yml"
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream, Loader=yaml.FullLoader)

url = "rtsp://admin:admin@123@103.99.13.188:80/cam/realmonitor?channel=1&subtype=0"
# Initialize camera capture
cap = cv2.VideoCapture(url)  # Use default camera (index 0)

# Check if the camera opened successfully 
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Define configuration parameters
config = {'text_overlay': True,
          'parking_overlay': True,
          'parking_id_overlay': True,
          'parking_detection': True,
          'min_area_motion_contour': 60,
          'park_sec_to_wait': 80}

# Function to rescale frame size
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
parking_status =  [False]*len(parking_data)
parking_buffer = [None] *len(parking_data)
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break

        frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
        frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()

        if config['parking_detection']:
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                rect = cv2.boundingRect(points)
                roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
                status = np.std(roi_gray) < 30 and np.mean(roi_gray) > 90
                if status != parking_status[ind] and parking_buffer[ind] == None:
                    parking_buffer[ind] = True
                elif status != parking_status[ind] and parking_buffer[ind] != None:
                    if parking_buffer[ind] == True:
                        parking_status[ind] = status
                        parking_buffer[ind] = None
                elif status == parking_status[ind] and parking_buffer[ind] != None:
                    parking_buffer[ind] = None

        if config['parking_overlay']:
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                if parking_status[ind]:
                    color = (0, 255, 0)  # green
                else:
                    color = (0, 0, 255)  # red
                cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=4, lineType=cv2.LINE_8)

        if config['text_overlay']:
            spot = sum(parking_status)
            occupied = len(parking_status) - spot
            str_on_frame = f"Free: {spot} Occupied: {occupied}"
            cv2.putText(frame_out, str_on_frame, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 128, 255), 5, cv2.LINE_AA)

        frame = rescale_frame(frame_out, percent=200)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    spot = sum(parking_status)
    vacancy =  len(parking_status) -spot
    return render_template('index.html' , vacancy =  vacancy)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
