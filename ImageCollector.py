import cv2
import os

#VideoSource or CCTV
cap = cv2.VideoCapture(0)

# FaceDetector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Provide a FolderPath for the images to be stored
outputFolder = 'ImageGallery'
os.makedirs(outputFolder, exist_ok=True)

# Capture and save up to 200 images
MaxImages = 200
image_count = 0

# Enter the name of the Employee and the ID
employeeName = input("Enter the employee's name: ")
employeeID = input("Enter the employee's ID: ")

# Program makes a subfolder for the employee's images with his ID
employeeFolder = os.path.join(outputFolder, f'{employeeID}-{employeeName}')
os.makedirs(employeeFolder, exist_ok=True)

while image_count < MaxImages:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Save images with employee's name
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = frame[y:y+h, x:x+w]  # Extract the region of interest (ROI)
        resized_face = cv2.resize(face_roi, (300, 300))

        # Save the resized image with employee's name
        filename = f'{employeeFolder}/{employeeName}_{image_count}.jpg'
        cv2.imwrite(filename, resized_face)

        # Rectangle around the employee's face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        image_count += 1

    # Display Frame
    cv2.imshow('Frame', frame)

    # Break if 200 images are saved
    if image_count >= MaxImages:
        print(f'Captured {MaxImages} images for {employeeName}.')
        break

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
    


