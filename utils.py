import cv2
import mediapipe as mp
import numpy as np
import utils

def extract_frame_from_video(path):
	"""
	:param path: path to the video
	return: list of numpy array video frame
	"""
	vid = cv2.VideoCapture(path)
	count = 0
	ret =1 
	frames=[]
	while ret:
		ret,frame = vid.read()
		if frame is not None and len(frame)>0:
			frames.append(frame)
			count = count + 1
	print(f"Extract total {count} frames from video")
	return frames

def apply_mask(img, mask):
	"""
	:param img: max 3 channel image
	:param mask: [0-255] values in mask
	return: image with mask apply
	"""
	masked_img= cv2.bitwise_and(img, img, mask=mask)
	return masked_img

def get_point_index(point, landmark_points):
	# return: the index of the point in the list
	return landmark_points.index(point)

def get_visuable_landmark(convexHull, landmark_points, xyz_landmark_points):
    """
    :param convexHull: convexhull of points
    :param landmark_points: points in uv_dimensional
    :param xyz_landmark_points: points in xyz_dimensional
    return mask matrix
    """
    #get z_coordination of convexHull 
    z=[]
    for point in convexHull:   
        indx=utils.get_point_index(tuple(point[0]), landmark_points)
        z.append(xyz_landmark_points[indx][2])
    
    visuable= np.ones(len(xyz_landmark_points), dtype=bool)
    #filter the hidden landmark 
    for idx, landmark in enumerate(xyz_landmark_points):
        if landmark[2] > np.max(z):
            visuable[idx]=False
    return visuable

def get_triangles(convexhull, landmarks_points, xyz_landmark_points):
	rect= cv2.boundingRect(convexhull)
	subdiv= cv2.Subdiv2D(rect)
	visuable=get_visuable_landmark(convexhull, landmarks_points, xyz_landmark_points)
	facial_landmarks= [point for idx, point in enumerate(landmarks_points) if visuable[idx]==1]
	subdiv.insert(facial_landmarks)
	triangles= subdiv.getTriangleList()
	triangles= np.array(triangles, np.int32)
	index_points_triangles= []

	for tri in triangles:
		pt1= (tri[0], tri[1])
		pt2= (tri[2], tri[3])
		pt3= (tri[4], tri[5])

		index_pt1=get_point_index(pt1, landmarks_points)
		index_pt2=get_point_index(pt2, landmarks_points)
		index_pt3=get_point_index(pt3, landmarks_points)

		triangle = [index_pt1, index_pt2, index_pt3]
		index_points_triangles.append(triangle)
	return index_points_triangles

def triangulation(triangle_point_index, landmark_points, img= None):
		#get triangluation point
		pt1= landmark_points[triangle_point_index[0]]
		pt2= landmark_points[triangle_point_index[1]]
		pt3= landmark_points[triangle_point_index[2]]

		triangle=np.array([pt1, pt2, pt3])
		rect= cv2.boundingRect(triangle)

		(x, y, w, h) = rect

		cropped_triangle = None

		if img is not None:
			cropped_triangle= img[y:y+h, x:x+w]

		cropped_triangle_mask= np.zeros((h,w), np.uint8)

		points=np.array([[pt1[0] - x, pt1[1] - y],
						 [pt2[0] - x, pt2[1] - y],
						 [pt3[0] - x, pt3[1] - y]])

		cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

		return points, cropped_triangle, cropped_triangle_mask, rect

def warp_triangle(points_target, points_dest, cropped_triangle_target, cropped_triangle_mask_dest, rect):
	
	(x, y, w, h) = rect
	M = cv2.getAffineTransform(np.float32(points_target),np.float32(points_dest))
	warped_triangle= cv2.warpAffine(cropped_triangle_target, M, (w,h))
	warped_triangle= apply_mask(warped_triangle, cropped_triangle_mask_dest)
	return warped_triangle.astype(np.uint8)

def getCenter(convexHull_points):
	"""
	return: the centre point of convexHull
	"""
	x1, y1, w1, h1 = cv2.boundingRect(convexHull_points)
	center = ((x1 + int(w1 / 2), y1 + int(h1 / 2)))
	#bounding_rectangle = cv2.rectangle(face2.copy(), (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
	return center

def getCenter_xyz(points):
	"""
	return: the centre point of each points in xyz_dimensional
	"""
	return np.mean(points,axis=0)

def AngleOfDepression(pointA, pointB):
	"""
	:point: in xyz_dimensional
	return: angle of 2 point. Its real part is in [-pi/2, pi/2]
	"""
	(xA, yA, zA)= pointA
	(xB, yB, zB)= pointB
	horizontal_dist = np.linalg.norm(np.array(xA,yA) - np.array(xB, yB))
	vertizontal_dist= zA- zB
	
	return np.arctan(vertizontal_dist/horizontal_dist) 
