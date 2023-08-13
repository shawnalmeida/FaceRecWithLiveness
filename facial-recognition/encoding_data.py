from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataset', required=True, help='Path to the dataset of input images for training')
parser.add_argument('-e', '--encoding', required=True, help='Path to store pickle file')
parser.add_argument('-d', '--detection', type=str, default='cnn', help='Face Detection model to use:hog or cnn')
args = vars(parser.parse_args())

print('STARTING THE PROGRAM')
Path_of_Images = list(paths.list_images(args['dataset']))

KnownEncodings = list()
KnownNames = list()

## Please Note that the images of the employees need to be showing their whole head in the picture and not just their face.
## Categorize each employee with their own folder and use atleast 7-10 images of each person minimum, and ideally around 25-30
## Images Dataset path should be dataset/EmployeeName/images.jpg or png

for (i, image_path) in enumerate(Path_of_Images):
    print(f'Processing Images... {i+1}/{len(Path_of_Images)}')
    name = image_path.split(os.path.sep)[-2]

    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args['detection'])
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        KnownEncodings.append(encoding)
        KnownNames.append(name)

print('Saving Encodings...')
data = {'encodings': KnownEncodings, 'names': KnownNames}
f = open(args['encoding'], 'wb')
f.write(pickle.dumps(data))
f.close()


