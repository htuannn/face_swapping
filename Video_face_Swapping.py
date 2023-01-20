import cv2
import mediapipe as mp
import numpy as np
import face_utils
import utils
import ImageProcessing
import glob

def main():
	path="Target/target.mp4"
	#img_path="Target/Timothee.jpeg"
	#target_imgs= [cv2.imread(img_path)]

	#target_imgs= [cv2.imread(file) for file in glob.glob("Target/*.jpg")]

	target_imgs = utils.extract_frame_from_video(path)
	targets_xyz_landmark_points=[]
	targets_landmark_points=[]
	for target_img in target_imgs:
		xyz_landmark_points, landmark_points = face_utils.get_face_landmark(target_img)
		targets_xyz_landmark_points.append(xyz_landmark_points)
		targets_landmark_points.append(landmark_points)

	targets_facial_angle=[]
	for target_xyz_landmark_points in targets_xyz_landmark_points:
		target_left_iris=face_utils.get_iris_landmark(target_xyz_landmark_points, return_xyz=True)
		target_right_iris=face_utils.get_iris_landmark(target_xyz_landmark_points, return_xyz=True, location='Right')
		target_facial_angle=utils.AngleOfDepression(utils.getCenter_xyz(target_left_iris)[0], utils.getCenter_xyz(target_right_iris)[0])
		targets_facial_angle.append(target_facial_angle)

	vid= cv2.VideoCapture("Destination/Mark-vid-input.mp4")
	ret =1 
	#video write
	frame_width = int(vid.get(3))
	frame_height = int(vid.get(4))

	size = (frame_width, frame_height)
	
	# Below VideoWriter object will create
	# a frame of above defined The output 
	# is stored in 'filename.avi' file.
	#result = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'XVID'),30, size)

	while (ret):
		ret,frame = vid.read()
		if ret==True:
			
			dest_img = frame.copy()
			face_landmark=face_utils.get_face_landmark(dest_img)
			if face_landmark is None:
				#result.write(frame)
				cv2.imshow("Result", frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				continue
			dest_xyz_landmark_points, dest_landmark_points = face_landmark
			dest_left_iris=face_utils.get_iris_landmark(dest_xyz_landmark_points, return_xyz=True)
			dest_right_iris=face_utils.get_iris_landmark(dest_xyz_landmark_points, return_xyz=True, location='Right')
		
			dest_facial_angle=utils.AngleOfDepression(utils.getCenter_xyz(dest_left_iris)[0], utils.getCenter_xyz(dest_right_iris)[0])
			most_simmilar_facial_angle= np.argmin(abs(targets_facial_angle - dest_facial_angle))
			target_img=target_imgs[most_simmilar_facial_angle]
		
			if dest_landmark_points is None or len(dest_landmark_points) < 478:
				continue

			dest_convexhull= cv2.convexHull(np.array(dest_landmark_points))

			#target_img=ImageProcessing.hist_match(target_img,dest_img)
			
			target_landmark_points= targets_landmark_points[most_simmilar_facial_angle]
			target_convexhull= cv2.convexHull(np.array(target_landmark_points))

			new_face, frame= face_utils.face_swapping(dest_img, dest_landmark_points, dest_xyz_landmark_points, dest_convexhull, target_img, target_landmark_points, target_convexhull, return_face= True)

			height, width, _ = frame.shape
			h, w, _ = target_img.shape
			rate= width/w
			#cv2.imshow("Target image", cv2.resize(target_img, (int(w * rate), int(h * rate))))
			#cv2.imshow("New face", new_face)
			cv2.imshow("Result", frame)
			#result.write(frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				 break
	vid.release()
	#result.release()

	print("done!!")
if __name__ == "__main__":
	main()