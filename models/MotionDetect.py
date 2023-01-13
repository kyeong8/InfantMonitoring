import imutils
import cv2
import numpy as np
import time

from collections import deque
from fcm import sendMessage

def motionDetect(frameQueue) :
    # print('====================Motion Detect Process Start====================')
    # =============================================================================
    # USER-SET PARAMETERS
    # =============================================================================

    # Number of frames to pass before changing the frame to compare the current
    # frame against
    FRAMES_TO_PERSIST = 10

    # Minimum boxed area for a detected motion to count as actual motion
    # Use to filter out noise or small objects
    MIN_SIZE_FOR_MOVEMENT = 2000

    # =============================================================================
    # CORE PROGRAM
    # =============================================================================

    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    dq = deque()
    next_block_flag = False
    start_time = time.time()
    messageCheck = False
    # LOOP!
    while True:
        # Set transient motion detected as false
        transient_movement_flag = False
        block_movement_flag = False

        if next_block_flag:
            start_time = time.time()
            next_block_flag = False
        
        # Read frame
        frame = frameQueue.get() # <===============================================================
        text = "Unoccupied"
        # Resize and save a greyscale version of the image
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if first_frame is None: first_frame = gray

        delay_counter += 1

        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = next_frame

        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True

                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if time.time() - start_time > 2:
            if transient_movement_flag == True:
                block_movement_flag = True
            if  len(dq) == 3:
                dq.popleft()

            dq.append(block_movement_flag)
            print('FIFO', dq)
            next_block_flag = True

        if sum(dq) == 3:
            text = "Frequently Movement Detected"
            if not messageCheck :
                messageCheck = True
                sendMessage('Frequently Moving Detected', 'Baby is moving now.')
        else:
            messageCheck = False
            text = "No Movement Detected"
        cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

        # ======================================TEST SHOW======================================
        # Splice the two video frames together to make one long horizontal one
        cv2.imshow("frame", np.hstack((frame_delta, frame)))

        # Interrupt trigger by pressing q to quit the open CV program
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break
