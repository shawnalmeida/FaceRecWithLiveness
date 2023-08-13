import os
from imutils.video import VideoStream
import face_recognition
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2


def FaceRecogLiveness(model_Path, le_Path, detector_folder, encodings, confidence=0.5):
    args = {'model': model_Path, 'le': le_Path, 'detector': detector_folder,
            'encodings': encodings, 'confidence': confidence}

    # Loading the encodings file (ENCODING DATA READING BLOCK)
    print('==> Loading the Encodings file..')
    with open(args['encodings'], 'rb') as file:
        encoded_data = pickle.loads(file.read())

    ## Loading the face detector from folder (DETECTOR IMPORTING)
    print('==> Loading the Face Detector..')
    protoP = os.path.sep.join([args['detector'], 'deploy.prototxt'])
    model_Path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
    detector_Net = cv2.dnn.readNetFromCaffe(protoP, model_Path)

    ## Loading the trained LivenessDetectorModel (LIVENESSDET BLOCK)
    liveness_model = tf.keras.models.load_model(args['model'])
    le = pickle.loads(open(args['le'], 'rb').read())

    ## Initializing the VidStream
    print('[Turning on Camera. Please Focus]>>>')
    cap = VideoStream(src=0).start()
    time.sleep(2)  # Giving the camera time to start

    sequence_count = 0  ## Keeping a count of a particular face being detected back to back

    name = 'Unknown'
    label_Name = 'fake'

    ## Working on the VideoStream frames being captured
    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=600)  # Resizing to make the process a bit smooth
        cv2.putText(frame, "Press 'q' to quit", (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
        ### Grabbing the frame dimensions and converting it into a blob
        ### (104.0, 177.0, 123.0) is the mean of image in FaceNet

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        ### Passing the blob to ze network to get predictions and detections
        detector_Net.setInput(blob)
        detections = detector_Net.forward()

        ### Iterating over the detections
        for i in range(0, detections.shape[2]):
            ## Extracting the confidence associated with the predictions
            confidence = detections[0, 0, i, 2]

            ## Filtering out weak detections
            if confidence > args['confidence']:
                ## Computing the (x, y) coordinates of the bounding box for ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                ## Expanding the bounding box a bit for better face rec
                startX = max(0, startX - 20)
                startY = max(0, startY - 20)
                endX = min(w, endX + 20)
                endY = min(h, endY + 20)

                ## Extracting face ROI and then preprocessing it like the training data
                face = frame[startY:endY, startX:endX]  ## Frame of Liveness Detection
                ## Expanding the bounding box for the model to classify easier
                face_to_rec = face  ## Face section for recognition
                ### Sorting out the exception of the face not facing the camera
                try:
                    face = cv2.resize(face, (32, 32))  ## Our Liveness model expects a 32x32 input
                except:
                    break

##########################################################################################################################

                # [FACE RECOGNITION BLOCK]
                rgb = cv2.cvtColor(face_to_rec, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)

                ## This name will be shown if the face isn't recognized
                name = 'Unknown'

                ## Iterating over the encoded faces in the pickle folder to see if exists
                for encoding in encodings:
                    match = face_recognition.compare_faces(encoded_data['encodings'], encoding)
                    if True in match:
                        ## Finding the indices of matched faces and then making a dictionary
                        match_indices = [i for i, b in enumerate(match) if b]
                        counts = {}

                        ## Looping over the matched indices and counting
                        for i in match_indices:
                            name = encoded_data['names'][i]
                            counts[name] = counts.get(name, 0) + 1

                        ## Getting the name with the highest count
                        name = max(counts, key=counts.get)

                face = face.astype('float') / 255.0
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)

                ## Passing the face ROI through the trained liveness detection model.
                ## Checking whether face is 'real' or 'fake'

                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_Name = le.classes_[j]  ## Getting Label of the predicted class

                ## Showing the labeling with bounding box on the frame
                label = f'{label_Name}: {preds[j]:.2f}'
                if name == 'Unknown' or label_Name == 'fake':
                    sequence_count = 0
                else:
                    sequence_count += 1
                print(f'{name}, {label_Name}, seq:{sequence_count}')

                if label_Name == 'fake':
                    cv2.putText(frame, "Don't try to cheat!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(frame, name, (startX, startY-35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                    cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                else:
                    cv2.putText(frame, name, (startX, startY - 35),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

        ## Output Frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        ## If 'q' is pressed, stop the loop
        if key == ord('q') or sequence_count == 20:
            break


    ## Cleaning up behind us
    cap.stop()
    cv2.destroyAllWindows()

    time.sleep(2)
    return name, label_Name

if __name__ == '__main__':
    name, label_Name = FaceRecogLiveness('liveness.model', 'label_encoder.pickle', 'face_detector',
                                         'facial-recognition/encoded_faces.pickle', confidence=0.5)

    print(name, label_Name)