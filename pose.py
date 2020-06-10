# %%
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import tensorflow as tf
import deepgaze
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

print(dlib.DLIB_USE_CUDA)
# %%
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# %%
sess = tf.Session()
head_pose_estimator = CnnHeadPoseEstimator(sess)
head_pose_estimator.load_pitch_variables('./deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf')
head_pose_estimator.load_yaw_variables('./deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf')
head_pose_estimator.load_roll_variables('./deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf')

# %%
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (fh, fw) = frame.shape[:2]

    if not ret:
        break

    faces = detector(gray, 0)

    for face in faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image = frame[y:y + h, x:x + w]
        image = cv2.resize(image, (480,480))
        font = cv2.FONT_HERSHEY_DUPLEX

        pitch = head_pose_estimator.return_pitch(image,radians=True)[0][0][0]
        yaw = head_pose_estimator.return_yaw(image,radians=True)[0][0][0]
        roll = head_pose_estimator.return_roll(image,radians=True)[0][0][0]
        
        #print 'data points ', 'pitch : ', pitch, ' roll ', roll, ' yaw ', yaw
        
        FONT = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'pitch = {:.2f}'.format(pitch), (20,25), FONT, 0.7, (0,255,0), 1)
        cv2.putText(frame, 'roll = {:.2f}'.format(roll), (20,50), FONT, 0.7, (0,255,0), 1)
        cv2.putText(frame, 'yaw = {:.2f}'.format(yaw), (20,75), FONT, 0.7, (0,255,0), 1)

        if pitch < -0.15 or pitch > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if yaw < -0.5 or yaw > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('pose', frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# %%
cap.release()
cv2.destroyAllWindows()   

# %%
