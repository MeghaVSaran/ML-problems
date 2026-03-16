#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import matplotlib.pyplot as plt

def extract_and_show_frames(video_path, num_frames=60):
    # Open the video file
    video_capture = cv2.VideoCapture('chakarboard.mp4')
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {'chakarboard.mp4'}")
        return
    
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames to extract
    interval = total_frames // num_frames
    
    current_frame = 0
    extracted_frames = 0
    
    # Create the directory to save frames if it doesn't exist
    if not os.path.exists('FRAMES'):
        os.makedirs('FRAMES')
    
    plt.figure(figsize=(20, 5))
    
    while extracted_frames < num_frames:
        # Set the video position to the current frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read the frame
        success, frame = video_capture.read()
        
        if not success:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the frame as an image file
        frame_filename = f'FRAMES/frame_{extracted_frames:03d}.png'
        cv2.imwrite(frame_filename, frame)
        
        # Display the frame using Matplotlib
        plt.subplot(1, num_frames, extracted_frames + 1)
        plt.imshow(frame_rgb)
        plt.axis('off')
        
        extracted_frames += 1
        current_frame += interval
    
    plt.tight_layout()
    plt.show()
    
    # Release the video capture object
    video_capture.release()

# Path to the video file
video_path = 'chakarboard.mp4'

# Extract and show frames
extract_and_show_frames('chakarboard.mp4')


# In[24]:


import numpy as np
import cv2
import glob

# Termination criteria for corner subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the known pattern (e.g., 7x7 chessboard)
chessboard_size = (7, 7)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Read all images from the specified folder
images = glob.glob('FRAMES/*.png')

# Counter for processed images
num_images_processed = 0

for fname in images:
    if num_images_processed >= 30:
        break
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(100)
        
        num_images_processed += 1

cv2.destroyAllWindows()

# Check if any chessboard corners were found
if not objpoints or not imgpoints:
    print("Error: No chessboard corners were found in any image.")
else:
    # Perform camera calibration to get the camera matrix, distortion coefficients, rotation, and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the camera calibration matrix
    print("Camera Calibration Matrix (Intrinsic Parameters):\n", mtx)
    print("Distortion Coefficients:\n", dist)
    
    # Print rotation vectors
    print("Rotation vectors:")
    for rvec in rvecs:
        print(rvec)
    
    # Print translation vectors
    print("Translation vectors:")
    for tvec in tvecs:
        print(tvec)

    # Save the calibration results for later use
    np.savez('camera_calibration_data.npz', camera_matrix=mtx, dist_coeffs=dist)


# In[26]:





# In[ ]:




