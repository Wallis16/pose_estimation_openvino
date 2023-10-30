from flask import Flask, render_template, Response
import cv2
import numpy as np
from estimator import PoseEstimator

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def check_area(poses, pts):
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]

        for _, (p, v) in enumerate(zip(points, points_scores)):
            if v > 0.1:
                point = Point(p)
                polygon = Polygon(pts)

                if polygon.contains(point) == True:
                    return polygon.contains(point)
                
def gen_frames():  # generate frame by frame from camera
    estimator = PoseEstimator(
        device_name="CPU", precision="FP16-INT8", video_url=""
    )
    while True:
        frame, scores, poses = estimator.get_frame()

        pts = np.array([[30, 50], [50, 200], 
            [200, 230], [200, 50]],
            np.int32)

        if check_area(poses, pts):
            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            color = (0, 0, 255)
            thickness = 2
            frame = cv2.polylines(frame, [pts], 
                                isClosed, color, thickness)

        else:
            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            color = (255, 0, 0)
            thickness = 2
            frame = cv2.polylines(frame, [pts], 
                                isClosed, color, thickness)

        success = True
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)