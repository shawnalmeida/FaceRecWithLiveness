import numpy as np
import argparse
import cv2
import os

# Constructing the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCVs deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Loading our Serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Open a pointer to the video file stream and initialize the total
# Number of Frames read and saved thus far
cap = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# Loop over frames from the video file stream
while True:

    # grab the frame from the file
    (grabbed, frame) = cap.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    read += 1
    if read % args["skip"] != 0:
        continue

    # Grab the frame dimensions and construct a glob from the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Ensure at least one face was found
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Ensure that the detection with the highest probability
        # Pass our minimum probability threshold (helping filter out some weak detections
        if confidence > args["confidence"]:
            # Compute the (x,y) coordinates of the bounding box for the face and extract face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # write the frame to the disk
            p = os.path.sep.join([args["output"], f"{saved}.png"])
            cv2.imwrite(p, face)
            saved += 1
            print(f"[INFO] saved {p} to disk")

# Cleaning Up
cap.release()
cv2.destroyAllWindows()