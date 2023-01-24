# Face-Swapping
![alt text](https://github.com/htuannn/Face-Swapping/samples.png "Samples")
Code for face swapping method preprocessing on image-to-image, video, realtime. 
This app is written by Python3 and uses OpenCV, and MediaPipe Face Mesh API and image blending to swap the face of a person in destination image/video or seen by camera with a face of a person in a provided target image/video.

## How to use it
To start the program you will have to run a Python file. You need to have Python3 and some additional libraries installed (Numpy, OpenCV-Python, Glob). 

You also have to install Mediapipe pakages from Google: https://google.github.io/mediapipe/getting_started/python.html

Mediapipe can be installed with the following command: 
```
pip install mediapipe
```

## Usage
Firstly, change the path to your own destination and target image/video in **_Face_Swapping.py source code file. 
For running the applications, run the following commands:
- Image swapping: 
```
python Image_Face_Swapping.py
```
- Video swapping: 
```
python Video_Face_Swapping.py
```
- Realtime swapping: 
```
python Realtime_Face_Swapping.py
```

## Releted
- [Face Swapping with OpenCV](https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
