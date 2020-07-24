from pose import HeadPoseEstimator

pose = HeadPoseEstimator()
pose.load_model()
pose.predict('img.jpg')
pose.save_data()
