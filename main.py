import os
import queue
import threading
import time
import pickle

import coloredlogs
import cv2
import numpy as np
import tensorflow as tf
import dlib
import imutils
from imutils import face_utils
import deepgaze
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import json
from img_json import im2json, json2im
import face_recognition

from GazeML.src.datasources import Video, Webcam
from GazeML.src.models import ELG
import GazeML.src.util.gaze

imgs_list = []
if os.path.isdir('img_cap') == False:
	os.mkdir('img_cap')

def gaze():
	
	coloredlogs.install(
		datefmt='%d/%m %H:%M',
		fmt='%(asctime)s %(levelname)s %(message)s',
		level='INFO',
	)

	with tf.Session() as session:

		# Declare some parameters
		batch_size = 2

		# Define webcam stream data source
		# Change data_format='NHWC' if not using CUDA
		data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
							 camera_id=0, fps=60,
							 data_format='NHWC',
							 eye_image_shape=(36, 60))

		model = ELG(
			session, train_data={'videostream': data_source},
			first_layer_stride=1,
			num_modules=2,
			num_feature_maps=32,
			learning_schedule=[
				{
					'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
				},
			],
		)


		# Begin visualization thread
		inferred_stuff_queue = queue.Queue()

		def _visualize_output():
			last_frame_index = 0
			last_frame_time = time.time()
			fps_history = []
			all_gaze_histories = [list() for _ in range(2)]
			gaze_history_max_len = 10
			prev = time.time() + 1
			i = 0

			while True:
				# If no output to visualize, show unannotated frame
				if inferred_stuff_queue.empty():
					next_frame_index = last_frame_index + 1
					if next_frame_index in data_source._frames:
						next_frame = data_source._frames[next_frame_index]
						if 'faces' in next_frame and len(next_frame['faces']) == 0:
							cv2.imshow('vis', next_frame['bgr'])
							last_frame_index = next_frame_index
					if cv2.waitKey(1) & 0xFF == ord('q'):
						cv2.destroyAllWindows()
						return
					continue

				# Get output from neural network and visualize
				output = inferred_stuff_queue.get()
				bgr = None
				for j in range(batch_size):
					frame_index = output['frame_index'][j]
					if frame_index not in data_source._frames:
						continue
					frame = data_source._frames[frame_index]
					if j == 0:
						img = frame['bgr'].copy()

					# Decide which landmarks are usable
					heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
					can_use_eye = np.all(heatmaps_amax > 0.7)
					can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
					can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

					start_time = time.time()
					eye_index = output['eye_index'][j]
					bgr = frame['bgr']
					eye = frame['eyes'][eye_index]
					eye_image = eye['image']
					eye_side = eye['side']
					eye_landmarks = output['landmarks'][j, :]
					eye_radius = output['radius'][j][0]
					if eye_side == 'left':
						eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
						eye_image = np.fliplr(eye_image)

					# Embed eye image and annotate for picture-in-picture
					eye_upscale = 2
					eye_image_raw = cv2.cvtColor(cv2.equalizeHist(eye_image), cv2.COLOR_GRAY2BGR)
					eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
					eye_image_annotated = np.copy(eye_image_raw)
					if can_use_eyelid:
						cv2.polylines(
							eye_image_annotated,
							[np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
																	 .reshape(-1, 1, 2)],
							isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA,
						)
					if can_use_iris:
						cv2.polylines(
							eye_image_annotated,
							[np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
																	  .reshape(-1, 1, 2)],
							isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)
						cv2.drawMarker(
							eye_image_annotated,
							tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
							color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)
					face_index = int(eye_index / 2)
					eh, ew, _ = eye_image_raw.shape
					v0 = face_index * 2 * eh
					v1 = v0 + eh
					v2 = v1 + eh
					u0 = 0 if eye_side == 'left' else ew
					u1 = u0 + ew
					bgr[v0:v1, u0:u1] = eye_image_raw
					bgr[v1:v2, u0:u1] = eye_image_annotated

					# Visualize preprocessing results
					frame_landmarks = (frame['smoothed_landmarks']
									   if 'smoothed_landmarks' in frame
									   else frame['landmarks'])
					for f, face in enumerate(frame['faces']):
						for landmark in frame_landmarks[f][:-1]:
							cv2.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
										  color=(0, 0, 255), markerType=cv2.MARKER_STAR,
										  markerSize=2, thickness=1, line_type=cv2.LINE_AA)
						cv2.rectangle(
							bgr, tuple(np.round(face[:2]).astype(np.int32)),
							tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
							color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)

					# Transform predictions
					eye_landmarks = np.concatenate([eye_landmarks,
													[[eye_landmarks[-1, 0] + eye_radius,
													  eye_landmarks[-1, 1]]]])
					eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
													   'constant', constant_values=1.0))
					eye_landmarks = (eye_landmarks *
									 eye['inv_landmarks_transform_mat'].T)[:, :2]
					eye_landmarks = np.asarray(eye_landmarks)
					eyelid_landmarks = eye_landmarks[0:8, :]
					iris_landmarks = eye_landmarks[8:16, :]
					iris_centre = eye_landmarks[16, :]
					eyeball_centre = eye_landmarks[17, :]
					eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
													eye_landmarks[17, :])

					# Smooth and visualize gaze direction
					num_total_eyes_in_frame = len(frame['eyes'])
					if len(all_gaze_histories) < num_total_eyes_in_frame:
						all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
					gaze_history = all_gaze_histories[eye_index]
					if can_use_eye:
						# Visualize landmarks
						cv2.drawMarker(  # Eyeball centre
							bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
							color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)
						i_x0, i_y0 = iris_centre
						e_x0, e_y0 = eyeball_centre
						theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
						phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
												-1.0, 1.0))
						current_gaze = np.array([theta, phi])
						gaze_history.append(current_gaze)
						if len(gaze_history) > gaze_history_max_len:
							gaze_history = gaze_history[-gaze_history_max_len:]
						GazeML.src.util.gaze.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
											length=120.0, thickness=1)
						if time.time() - prev > 1:
							gh = np.mean(gaze_history, axis=0)
							gh = np.round(gh, 3)
							print('{} pitch : {:6} yaw : {:6}'.format(j, gh[0], gh[1]))

					# else:
					# 	gaze_history.clear()

					if can_use_eyelid:
						cv2.polylines(
							bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
							isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA,
						)

					if can_use_iris:
						cv2.polylines(
							bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
							isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
						)
						cv2.drawMarker(
							bgr, tuple(np.round(iris_centre).astype(np.int32)),
							color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
							thickness=1, line_type=cv2.LINE_AA,
						)

					dtime = 1e3*(time.time() - start_time)
					if 'visualization' not in frame['time']:
						frame['time']['visualization'] = dtime
					else:
						frame['time']['visualization'] += dtime

					def _dtime(before_id, after_id):
						return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

					def _dstr(title, before_id, after_id):
						return '%s: %dms' % (title, _dtime(before_id, after_id))

					if eye_index == len(frame['eyes']) - 1:
						# Calculate timings
						frame['time']['after_visualization'] = time.time()
						fps = int(np.round(1.0 / (time.time() - last_frame_time)))
						fps_history.append(fps)
						if len(fps_history) > 60:
							fps_history = fps_history[-60:]
						fps_str = '%d FPS' % np.mean(fps_history)
						last_frame_time = time.time()
						fh, fw, _ = bgr.shape
						cv2.putText(bgr, fps_str, org=(fw - 110, fh - 20),
								   fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
								   color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
						cv2.putText(bgr, fps_str, org=(fw - 111, fh - 21),
								   fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.79,
								   color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
						cv2.imshow('vis', bgr)
						last_frame_index = frame_index

						# Quit?
						if cv2.waitKey(1) & 0xFF == ord('q'):
							cv2.destroyAllWindows()
							return

						# Print timings
						if frame_index % 60 == 0:
							latency = _dtime('before_frame_read', 'after_visualization')
							processing = _dtime('after_frame_read', 'after_visualization')
							timing_string = ', '.join([
								_dstr('read', 'before_frame_read', 'after_frame_read'),
								_dstr('preproc', 'after_frame_read', 'after_preprocessing'),
								'infer: %dms' % int(frame['time']['inference']),
								'vis: %dms' % int(frame['time']['visualization']),
								'proc: %dms' % processing,
								'latency: %dms' % latency,
							])
							# print('%08d [%s] %s' % (frame_index, fps_str, timing_string))

				ghl = all_gaze_histories[0]
				if len(ghl) > gaze_history_max_len:
					ghl = ghl[-gaze_history_max_len:]
				left = np.asarray(ghl)

				ghr = all_gaze_histories[1]
				if len(ghr) > gaze_history_max_len:
					ghr = ghr[-gaze_history_max_len:]
				right = np.asarray(ghr)

				if not left.any() or not right.any():
					continue

				if time.time() - prev > 1:
					i += 1
					prev = time.time()
					cv2.imwrite(os.getcwd() + os.sep + 'img_cap' + os.sep + f'{i}.jpg', img)

					jstr = im2json(img)
					img_dict = json.loads(jstr)
					img_dict['index'] = i
					img_dict['left_eye'] = {
						'pitch' : np.mean(left, axis=0)[0],
						'yaw' : np.mean(left, axis=0)[1]
					}
					img_dict['right_eye'] = {
						'pitch' : np.mean(right, axis=0)[0],
						'yaw' : np.mean(right, axis=0)[1]
					}

					imgs_list.append(img_dict)

		visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
		visualize_thread.daemon = True
		visualize_thread.start()

		# Do inference forever
		infer = model.inference_generator()
		while True:
			output = next(infer)
			for frame_index in np.unique(output['frame_index']):
				if frame_index not in data_source._frames:
					continue
				frame = data_source._frames[frame_index]
				if 'inference' in frame['time']:
					frame['time']['inference'] += output['inference_time']
				else:
					frame['time']['inference'] = output['inference_time']
			inferred_stuff_queue.put_nowait(output)

			if not visualize_thread.isAlive():
				break

			if not data_source._open:
				break

	cv2.destroyAllWindows()

	with open("gaze.json", "w") as p: 
		json.dump(imgs_list, p, indent = 4)


def pose():
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	sess = tf.Session()
	head_pose_estimator = CnnHeadPoseEstimator(sess)
	head_pose_estimator.load_pitch_variables('pitch.tf')
	head_pose_estimator.load_yaw_variables('yaw.tf')
	head_pose_estimator.load_roll_variables('roll.tf')


	with open("gaze.json", "r") as p: 
		data_list = json.load(p)

	for data in data_list:
		frame = json2im(json.dumps(data))
		frame = cv2.flip(frame, 1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(fh, fw) = frame.shape[:2]

		faces = detector(gray, 0)

		for face in faces:
			(x, y, w, h) = face_utils.rect_to_bb(face)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			image = frame[y:y + h, x:x + w]

			try:
				image = cv2.resize(image, (480,480))
			except:
				print('Exception')
				continue

			pitch = head_pose_estimator.return_pitch(image,radians=True)[0][0][0]
			yaw = head_pose_estimator.return_yaw(image,radians=True)[0][0][0]
			roll = head_pose_estimator.return_roll(image,radians=True)[0][0][0]
			
			print('data points ', 'pitch ', pitch, ' yaw ', yaw, ' roll ', roll)
			
			FONT = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, 'pitch = {:.2f}'.format(pitch), (20,25), FONT, 0.7, (0,255,0), 1)
			cv2.putText(frame, 'yaw = {:.2f}'.format(yaw), (20,75), FONT, 0.7, (0,255,0), 1)
			cv2.putText(frame, 'roll = {:.2f}'.format(roll), (20,50), FONT, 0.7, (0,255,0), 1)

			if pitch < -0.15 or pitch > 0:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			if yaw < -0.5 or yaw > 0.5:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

			data['pose'] = {
				'pitch' : float(pitch),
				'yaw' : float(yaw),
				'roll' : float(roll)
			}

		if not faces:
			data['pose'] = None

	cv2.destroyAllWindows()   

	with open("pose.json", "w") as p: 
		json.dump(data_list, p, indent = 4)


def face_rec():
	KNOWN_FACES_DIR = "known_faces"

	TOLERANCE = 0.45
	FRAME_THICKNESS = 3
	FONT_THICKNESS = 2
	MODEL = "cnn"

	print("Loading known faces")

	known_faces = []
	known_names = []

	def train():
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

	with open('known_faces.dat', 'rb') as f:
		known_faces = pickle.load(f)

	with open('known_names.dat', 'rb') as f:
		known_names = pickle.load(f)

	with open("gaze.json", "r") as p: 
		data_list = json.load(p)

	print("Processing unknown_faces")
	for data in data_list:
		image = json2im(json.dumps(data))

		locations = face_recognition.face_locations(image, model=MODEL)
		encodings = face_recognition.face_encodings(image, locations)

		for face_encoding, face_location in zip(encodings, locations):
			results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
			match = None
			if True in results:
				match = known_names[results.index(True)]
				print(f"Match found: {match}")
				top_left = (face_location[3], face_location[0])
				bottom_right = (face_location[1], face_location[2])
				color = [0, 255, 0]
				cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

				top_left = (face_location[3], face_location[2])
				bottom_right = (face_location[1], face_location[2]+22)
				cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
				cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
			data['name'] = match

	cv2.destroyAllWindows()

	with open("face_rec.json", "w") as p: 
		json.dump(data_list, p, indent = 4)


gaze()
pose()
face_rec()