# Face Recognition using dlib and face_recognition

This code demonstrates face recognition using the dlib library and the face_recognition module. It recognizes faces in an image and draws boxes around them, along with labels indicating the names of known individuals.

## Installation

Before running the code, make sure to install the required dependencies in the Colab environment:

```bash
# Update package list
!sudo apt-get -y update

# Install required dependencies
!sudo apt-get install -y --fix-missing \
    cmake \
    libgtk2.0-dev

# Clean up unnecessary files
!sudo apt-get clean && rm -rf /tmp/* /var/tmp/*

# Install dlib
!pip install dlib==19.9

# Install face_recognition
!pip install face_recognition
```

## Usage

Load the required libraries and display the image:
```bash
from PIL import Image, ImageDraw
from IPython.display import display
import face_recognition
import numpy as np

# Load image
pil_im = Image.open('two_people.jpg')
display(pil_im)
```
![two_people](https://github.com/CassioD/Find-faces-in-pictures/assets/87616806/eb6fd7d7-efc9-4b84-abe9-05fe685b33e1)

## Define known faces and their encodings:
```bash
# Load sample images and learn face encodings
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]
```

## Perform face recognition on an unknown image:

```bash
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("two_people.jpg")

# Find all faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
```

## Draw boxes and labels around recognized faces:

```bash
# Convert the image to a PIL-format image
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Check if the face matches any known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Use the known face with the smallest distance as the best match
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with the name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
```

## Display the resulting image:
```bash
# Display the resulting image
display(pil_image)
```
![image](https://github.com/CassioD/Find-faces-in-pictures/assets/87616806/21776356-f712-4537-a9ab-3994d4afc557)

## Note: Ensure that the sample images (obama.jpg, biden.jpg, two_people.jpg) are available in the working directory.




