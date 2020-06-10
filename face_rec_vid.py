import face_recognition
import os
import cv2
import pickle

KNOWN_FACES_DIR = "known_faces"

TOLERANCE = 0.4
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"

video = cv2.VideoCapture(0)

print("Loading known faces")

known_faces = []
known_names = []

'''
for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)
		if not encoding:
			continue
		else:
			encoding = encoding[0]
		known_faces.append(encoding)
		known_names.append(name)

with open('known_faces.dat', 'wb') as f:
    pickle.dump(known_faces, f)

with open('known_names.dat', 'wb') as f:
    pickle.dump(known_names, f)
'''

with open('known_faces.dat', 'rb') as f:
    known_faces = pickle.load(f)

with open('known_names.dat', 'rb') as f:
    known_names = pickle.load(f)

def findname(results, known_names):
	count = {}
	total = {}
	prob = {}

	for name in known_names:
		total[name] = count[name] = prob[name] = 0

	for name, value in zip(known_names, results):
		total[name] += 1
		if value == True:
			count[name] += 1

	for name in known_names:
		prob[name] = count[name] / total[name]

	print(count)
	print(total)
	print(prob)

	return max([key for key in prob], key=(lambda key: prob[key]))

print("Processing unknown_faces")
while True:
	ret, image = video.read()

	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = face_recognition.face_encodings(image, locations)

	for face_encoding, face_location in zip(encodings, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			#match = findname(results, known_names)
			print(f"Match found: {match}")
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])
			color = [0, 255, 0]
			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

	cv2.imshow("filename", image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

video.release()
cv2.destroyAllWindows()