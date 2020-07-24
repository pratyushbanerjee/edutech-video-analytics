import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import cv2
import dlib
import json
from imutils import face_utils

from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

class HeadPoseEstimator:

	def __init__(self):

		self.sess = tf.Session()
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('models' + os.sep + 'shape_predictor_68_face_landmarks.dat')
		self.data = None

	def load_model(self):

		self.head_pose_estimator = CnnHeadPoseEstimator(self.sess)
		self.head_pose_estimator.load_pitch_variables('models' + os.sep + 'pitch.tf')
		self.head_pose_estimator.load_yaw_variables('models' + os.sep + 'yaw.tf')
		self.head_pose_estimator.load_roll_variables('models' + os.sep + 'roll.tf')

	def predict(self, img):

		print('Head pose estimation started')

		frame = cv2.imread(img)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = self.detector(gray, 0)

		for face in faces:
			(x, y, w, h) = face_utils.rect_to_bb(face)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			image = frame[y:y + h, x:x + w]

			try:
				image = cv2.resize(image, (480,480))
			except:
				print('Exception')
				continue

			pitch = self.head_pose_estimator.return_pitch(image,radians=True)[0][0][0]
			yaw = self.head_pose_estimator.return_yaw(image,radians=True)[0][0][0]
			roll = self.head_pose_estimator.return_roll(image,radians=True)[0][0][0]
			
			FONT = cv2.FONT_HERSHEY_DUPLEX
			print('data points ', 'pitch ', pitch, ' roll ', roll, ' yaw ', yaw)
        
			cv2.putText(frame, f'pitch = {pitch:.2f}', (20,25), FONT, 0.7, (0,255,0), 1)
			cv2.putText(frame, f'roll = {roll:.2f}', (20,50), FONT, 0.7, (0,255,0), 1)
			cv2.putText(frame, f'yaw = {yaw:.2f}', (20,75), FONT, 0.7, (0,255,0), 1)

			if pitch < -0.15 or pitch > 0:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			if yaw < -0.5 or yaw > 0.5:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

			cv2.imshow('pose', frame)    
			cv2.waitKey(0)

			self.data = {
				'pitch' : float(pitch),
				'yaw' : float(yaw),
				'roll' : float(roll)
			}
	
		print('Head pose estimation complete')
	
	def save_data(self):

		with open('pose.json', 'w') as p: 
			json.dump(self.data, p, indent = 4)

		print('Data saved as pose.json')