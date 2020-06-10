import cv2
import os
import time

cap = cv2.VideoCapture(0)

if os.path.isdir('capture') == False:
	os.mkdir('capture')

fps = 1
prev = time.time()

while True:
	ret, frame = cap.read()
	time_elapsed = time.time() - prev

	if time_elapsed >= 1.0 / fps:
		prev = time.time()
		cv2.imwrite('capture' + os.sep + str(time.time()) + '.jpg', frame)
	
	cv2.imshow('Capture', frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
