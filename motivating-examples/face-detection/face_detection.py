import cv2

PATH_TO_IMAGES = './pictures'
PATH_TO_RESOURCES = './resources'
INPUT_FILES = ['kristina.jpg']
PATH_TO_CASCADE_XML = '/'.join((PATH_TO_RESOURCES, 'haarcascade_frontalface_default.xml'))

for file in INPUT_FILES:
    file_name = "/".join((PATH_TO_IMAGES, file))

    # Read the input image
    image = cv2.imread(file_name)

    # Create the cascade
    face_cascade = cv2.CascadeClassifier(PATH_TO_CASCADE_XML)

    # Use grayscale picture
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.equalizeHist(grayimg)

    # Run the classifier
    faces = face_cascade.detectMultiScale(image=grayimg,
                                          scaleFactor=1.15,
                                          minNeighbors=10,
                                          minSize=(30, 30))

    print("Processing file %s" % file)

    num_of_faces=len(faces)
    if num_of_faces!=0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("%s faces detected" % num_of_faces)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
    else:
        print("no faces detected")
