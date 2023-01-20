import cv2
import mediapipe as mp
import numpy as np
import utils 
import ImageProcessing
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

mouth_landmark_index=[13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78, 191, 80, 81, 82]


def get_face_landmark(img):
	with mp_face_mesh.FaceMesh(static_image_mode = True,	 
								refine_landmarks=True,
								max_num_faces=1) as face_mesh:
		results= face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		if results.multi_face_landmarks is None:
			return None
		if len(results.multi_face_landmarks) > 1:
			print("There are too much face")

		xyz_landmark_points=[]
		landmark_points= []

		face_landmark= results.multi_face_landmarks[0].landmark
		for landmark in face_landmark:
			x = landmark.x
			y = landmark.y
			z = landmark.z
			relative_x=int(x * img.shape[1])
			relative_y=int(y * img.shape[0])
			#cv2.circle(img, (relative_x,relative_y) , 3, (255,255,255), -1)
			xyz_landmark_points.append((x, y, z))
			landmark_points.append((relative_x,relative_y))

	return xyz_landmark_points, landmark_points

def get_iris_landmark(landmark_points, return_xyz=True, location= "Left"):
	points= []
	if location == 'Left':
		iris_landmark_index= mp_face_mesh.FACEMESH_LEFT_IRIS
	else:
		iris_landmark_index= mp_face_mesh.FACEMESH_RIGHT_IRIS
	for idx, _ in iris_landmark_index:
		source = landmark_points[idx]
		if return_xyz:
			relative_source = [[source[0], source[1], source[2]]]
		else: 
			relative_source = [[int(source[0]), int(source[1])]]
		points.append(relative_source)
	return np.array(points)

def get_eye_landmark(landmark_points, location='Left'):
	points= []
	if location == 'Left':
		eye_landmark_index= mp_face_mesh.FACEMESH_LEFT_EYE
	else:
		
		eye_landmark_index= mp_face_mesh.FACEMESH_RIGHT_EYE
	for idx, _ in eye_landmark_index:
		source = landmark_points[idx]
		relative_source = [[int(source[0]), int(source[1])]]
		points.append(relative_source)
	return np.array(points)

def get_mouth_landmark(landmark_points):
	points= []
	for idx in mouth_landmark_index:
		source = landmark_points[idx]
		relative_source = [[int(source[0]), int(source[1])]]
		points.append(relative_source)
	return np.array(points)

def protected(mask, landmark_points):
	convexhull_landmark_points= cv2.convexHull(landmark_points)
	mask=cv2.fillConvexPoly(mask, np.array(convexhull_landmark_points), 0)
	return mask

def create_face_mask(img, img_convexhull, img_landmark_points, protected_eyes= False, protected_mouth= False):
	face_mask = np.zeros(img.shape[:2])
	face_mask = cv2.fillConvexPoly(face_mask, img_convexhull, 255)
	if protected_eyes:
		face_mask = protected(face_mask, get_eye_landmark(img_landmark_points))
		face_mask = protected(face_mask, get_eye_landmark(img_landmark_points, location="Right"))
	if protected_mouth:
		face_mask = protected(face_mask, get_mouth_landmark(img_landmark_points))
	return face_mask.astype(np.uint8)

def face_swapping(dest_img, dest_landmark_points, dest_xyz_landmark_points, dest_convexhull, target_img, target_landmark_points, target_convexhull, return_face=False):
	new_face = np.zeros_like(dest_img, np.uint8)

	for triangle_index in utils.get_triangles(dest_convexhull, dest_landmark_points, dest_xyz_landmark_points):
	
		points_dest, _ ,cropped_triangle_mask_dest, rect =utils.triangulation(triangle_index, dest_landmark_points)


		points_target, cropped_triangle_target, cropped_triangle_mask_target, _ =utils.triangulation(triangle_index, target_landmark_points, target_img)

		#warp triangles
		warped_triangle = utils.warp_triangle(points_target, points_dest, cropped_triangle_target, cropped_triangle_mask_dest, rect)
		(x, y, w, h)= rect

		triangle_area= new_face[y: y + h, x: x + w]

		#remove the line between the triangles
		triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)
		_, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
		warped_triangle = utils.apply_mask(warped_triangle, mask_triangles_designed)
		triangle_area= cv2.add(triangle_area, warped_triangle)

		new_face[y: y + h, x: x + w]= triangle_area

	dest_mask = create_face_mask(dest_img, dest_convexhull, dest_landmark_points, protected_eyes=True, protected_mouth=True)
	dest_without_face= utils.apply_mask(dest_img, cv2.bitwise_not(dest_mask))

	#smoothing new face
	new_face = cv2.medianBlur(new_face, 3)

	new_face= utils.apply_mask(new_face, dest_mask)

	old_face= utils.apply_mask(dest_img, dest_mask)
	blending_mask= create_face_mask(dest_img, dest_convexhull, dest_landmark_points)

	cv2.GaussianBlur(blending_mask, (51, 51), 30, dst=blending_mask)
	blending_mask = utils.apply_mask(blending_mask, dest_mask)
	target_face= ImageProcessing.blend_with_mask_matrix(new_face, old_face, blending_mask)

	result = cv2.add(dest_without_face, target_face)
	result=cv2.seamlessClone(result, dest_img, blending_mask, utils.getCenter(dest_convexhull), cv2.NORMAL_CLONE)
	if return_face:
		return target_face, result
	return result