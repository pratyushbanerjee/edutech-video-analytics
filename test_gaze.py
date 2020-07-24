from gaze import GazeAngleEstimator

pose = GazeAngleEstimator()
pose.predict('video.mp4')
pose.save_data()