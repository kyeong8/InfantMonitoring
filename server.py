from flask import Flask, render_template
from flask import request
from flask import Response
from flask import stream_with_context
from multiprocessing import Queue

import cv2
import dlib
import time
import queue
import imutils
import platform
import numpy as np

# AI models
from models.BlinkDetect import blinkDetect
from models.posenet import poseDetect # <=================================Jetson Environment======================================
from threading import Thread

# ====================전역 변수 선언====================
app = Flask(__name__)
capture = None
updateThread = None
readThread = None
width = 320
height = 240
cameraOn = False
streamQueueChecked = False
streamQueue = queue.Queue(maxsize=128)
Q = queue.Queue(maxsize=128)

poseEstimationChecked = False
frequentlyMoveChecked = False
blinkDetectionChecked = False

motionFrameQueue = Queue(maxsize=128)

# main page
@app.route('/')
def index():
    global streamQueueChecked
    global streamQueue

    # Empty the streamQueue if streamQueueChecked is True
    if streamQueueChecked :
        print('Stream Queue is Cleared')
        streamQueueChecked = False
        with streamQueue.mutex :
            streamQueue.queue.clear()
    
    return render_template('index.html', 
                            FaceCoverBlanketRemoveState='ON' if poseEstimationChecked else 'OFF', 
                            FrequentlyMoveState='ON' if frequentlyMoveChecked else 'OFF', 
                            AwakeState='ON' if blinkDetectionChecked else 'OFF')

# streaming page
@app.route('/stream_page')
def stream_page():
    global streamQueueChecked

    streamQueueChecked = True
    return render_template('stream.html', 
                            FaceCoverBlanketRemoveState='ON' if poseEstimationChecked else 'OFF', 
                            FrequentlyMoveState='ON' if frequentlyMoveChecked else 'OFF', 
                            AwakeState='ON' if blinkDetectionChecked else 'OFF')

# stream function
@app.route('/stream')
def stream() :
    try :
        return Response(
                            stream_with_context(stream_gen()),
                            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e :
        print('[Badger]', 'stream error : ', str(e))

# setting page
@app.route('/setting')
def setting():
    return render_template('setting.html')

# setting post function
@app.route('/setting_post', methods=['POST'])
def settingPost() :
    global poseEstimationChecked
    global frequentlyMoveChecked
    global blinkDetectionChecked
    
    if request.method == 'POST' :
        poseEstimationChecked = str(request.form.get('PoseEstimation')) == 'on'
        frequentlyMoveChecked = str(request.form.get('FrequentlyMove')) == 'on'
        blinkDetectionChecked = str(request.form.get('BlinkDetection')) == 'on'
        print('MODE : ', poseEstimationChecked, frequentlyMoveChecked, blinkDetectionChecked)

    return render_template('index.html', 
                            FaceCoverBlanketRemoveState='ON' if poseEstimationChecked else 'OFF', 
                            FrequentlyMoveState='ON' if frequentlyMoveChecked else 'OFF', 
                            AwakeState='ON' if blinkDetectionChecked else 'OFF')

# camera post function
@app.route('/camera_post', methods=['POST'])
def camerapost() :
    if request.method == 'POST' :
        on = str(request.form.get('CameraOn')) == 'on'
        off = str(request.form.get('CameraOff')) == 'off'
    if on and not cameraOn :
        print('========================================Camera ON=========================================')
        runCam(0)
    elif off and cameraOn :
        print('========================================Camera OFF========================================')
        stopCam()
    return render_template('index.html', 
                            FaceCoverBlanketRemoveState='ON' if poseEstimationChecked else 'OFF', 
                            FrequentlyMoveState='ON' if frequentlyMoveChecked else 'OFF', 
                            AwakeState='ON' if blinkDetectionChecked else 'OFF')

# ===========================================================================================================================
# ====================================================== Function Area ======================================================
# ===========================================================================================================================

# 웹페이지에 바이트 코드를 이미지로 출력하는 함수
def stream_gen() :
    while True :
        frame = bytescode()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 카메라 시작 함수
def runCam(src=0) :
    global capture
    global cameraOn
    global updateThread
    global readThread

    stopCam()
    if platform.system() == 'Windows' :        
        capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else :
        capture = cv2.VideoCapture(src)
    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if updateThread is None :
        print('Update Thread Start')
        updateThread = Thread(target=updateVideoFrame, args=(), daemon=False)
        updateThread.start()
    
    if readThread is None :
        print('Read Thread Start')
        readThread = Thread(target=readVideoFrame, args=(), daemon=False)
        readThread.start()
    cameraOn = True

# 카메라 중지 함수
def stopCam() :
    global cameraOn

    cameraOn = False

    if capture is not None :
        capture.release()
        clearVideoFrame()

# 영상 데이터를 실시간으로 Queue에 update하는 Thread 내용, 전역변수 cameraOn이 False면
# 빈 while문 진행
def updateVideoFrame() :
    while True :
        if cameraOn :
            (ret, frame) = capture.read()

            if ret :
                Q.put(frame)

                if streamQueueChecked :
                    streamQueue.put(frame)

                if frequentlyMoveChecked :
                    motionFrameQueue.put(frame)

                if blinkDetectionChecked :
                    blinkDetect(frame)

                if poseEstimationChecked :
                    poseDetect(frame)

# 영상 데이터를 실시간으로 Queue에서 read하는 Thread 내용, 전역변수 cameraOn이 False면
# 빈 while문 진행
def readVideoFrame() :
    while True :
        if cameraOn :
            frame = Q.get()

# Queue에 있는 영상 데이터를 삭제하는 함수
def clearVideoFrame() :
    with Q.mutex :
        Q.queue.clear()

# 검은화면을 출력하는 함수
def blankVideo() :
    return np.ones(shape=[height, width, 3], dtype=np.uint8)

# 이미지 데이터를 바이트 코드로 변환하는 함수
def bytescode() :
    frame = streamReadFrame()
    if capture is None or frame is None or not capture.isOpened():
        frame = blankVideo()
    else :
        frame = imutils.resize(frame, width=int(width))
    return cv2.imencode('.jpg', frame)[1].tobytes()

def streamReadFrame() :
    if cameraOn :
        return streamQueue.get()
    else :
        return None
