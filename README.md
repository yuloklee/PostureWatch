# PostureWatch
Detect posture and send desktop notifications to remind you when your posture is poor

## Setup and Usage
1. Clone the repository or download the source files
2. Run `python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel` in the root directory
3. Enter `R` to reset face reference position
4. Enter `ctrl + Q` to exit

## How it works
Posture watch uses OpenCV with caffe models that have been trained to detect faces. It takes the landmarks returned from the detection algorithm to find the centroid of a face, and keeps track of it for the duration of the program. If the program detects that the centroid location drops too low, it sends a notification to the user that they are slouching, where they can either sit up, or enter the `R` key to reset their position.
