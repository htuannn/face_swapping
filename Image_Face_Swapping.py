import cv2
import mediapipe as mp
import numpy as np
import face_utils
import ImageProcessing

def main(dest_img_path, target_img_path, result_path='result.jpg'):
	dest_img=cv2.imread(dest_img_path)
	target_img= cv2.imread(target_img_path)

	dest_xyz_landmark_points, dest_landmark_points= face_utils.get_face_landmark(dest_img)
	dest_convexhull= cv2.convexHull(np.array(dest_landmark_points))

	target_img_hist_match=ImageProcessing.hist_match(target_img,dest_img)
	
	_, target_landmark_points= face_utils.get_face_landmark(target_img)
	target_convexhull= cv2.convexHull(np.array(target_landmark_points))

	new_face, result= face_utils.face_swapping(dest_img, dest_landmark_points, dest_xyz_landmark_points, dest_convexhull, target_img, target_landmark_points, target_convexhull, return_face= True)

	height, width, _ = dest_img.shape
	h, w, _ = target_img.shape
	rate= width/w
	cv2.imshow("Destination image", dest_img)
	cv2.imshow("Target image", cv2.resize(target_img, (int(w * rate), int(h * rate))))
	cv2.imshow("New face", new_face)
	cv2.imshow("Result", result)
	cv2.imwrite(result_path, result)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	dest_img_path='sontung.webp'
	target_img_path='timothee.webp'
	result_path='result/sontungtimothee.jpg'
	main(dest_img_path, target_img_path, result_path)