import cv2
import numpy as np
import os

# Setting up the Video Capture
cap = cv2.VideoCapture('Resources/.mp4')
cap.set(3, 640)
cap.set(4, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, takes the person's name

face_name = input('\n Enter the person name to store: ')
print("\n [INFO] Initializing the Video Capture Procedure..")

# Initializing the face count to collect images
count = 0
while True:
    ret, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (250, 0, 0), 2)
        count += 1

        # Saving the Images in the dataset folder
        cv2.imwrite("dataset/"+str(face_name)+'.'+str(count)+".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('Image', image)
    n = cv2.waitKey(100) & 0xff  # 'ESCAPE Button'
    if n == 27:
        break
    elif count >= 80:  # Takes face samples
        break
print("\n [INFO] Finished the Process, Exiting Program")
cap.release()
cv2.destroyAllWindows()
    


