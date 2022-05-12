import cv2

from random import randrange


# loading pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# reading in a test image

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    # read curr frame
    successful_frame_read, frame = webcam.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_image)

    # draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256),
                                                      randrange(128, 256)), 10)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break


webcam.release()
