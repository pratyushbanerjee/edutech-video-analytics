from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

def bound(points):
	x = min([x for (x,y) in points])
	y = min([y for (x,y) in points])
	w = max([x for (x,y) in points]) - x
	h = max([y for (x,y) in points]) - y
	return (x, y, w, h)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray, 0)

	for face in faces:
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)

		right_eye = shape[36:42]
		left_eye = shape[42:48]

		(x, y, w, h) = bound(right_eye)
		cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
		# right = frame[y-10:y+h+10, x-10:x+w+10] 
		# right = imutils.resize(right, width=300)
		
		(x, y, w, h) = bound(left_eye)
		cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
		# left = frame[y-10:y+h+10, x-10:x+w+10]  
		# left = imutils.resize(left, width=300)

		# for (x, y) in left_eye:
		# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		# for (x, y) in right_eye:
		# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# for (x,y) in shape:
		# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), 5)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break
 
cv2.destroyAllWindows()
vs.stop()