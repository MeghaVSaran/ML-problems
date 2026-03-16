This code captures a video of cheque board and extracts frames from it(30). 
Then it determines the calibration matrix of the camera from which the video was taken.

How to run: open folder in vs code and click on'run below'

The frames of video are stored in a folder named 'FRAMES' in this folder.

Input is read from video_capture = cv2.VideoCapture('chakarboard.mp4')

O/P: Calibration matrix, rotation vector (printed in program)