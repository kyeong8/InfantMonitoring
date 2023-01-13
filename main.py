from server import app, motionFrameQueue
from models.MotionDetect import motionDetect
from multiprocessing import Process

# poseDetect rendering
import sys
from jetson_utils import videoOutput # <=================================Jetson Environment======================================

ouput = videoOutput("", argv=sys.argv) # <=================================Jetson Environment======================================

if __name__ == '__main__':
    print('CV on')

    # Create MotionDeetect Process
    motionProcess = Process(target=motionDetect, args=(motionFrameQueue,), daemon=False)
    motionProcess.start()

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    print('main close')