import cv2
import face_recognition
import tensorflow

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Frame not read properly.")
        break

    # Resize the frame for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect face locations
    loc = face_recognition.face_locations(small_frame, model='cnn')

    # Draw rectangles around detected faces
    for (top, right, bottom, left) in loc:
        # Scale back up the face locations since the frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cp = frame[top:bottom:,left:right]
        cp = cv2.GaussianBlur(cp, (99,99), 10)
        frame[top:bottom:,left:right] = cp
        # Draw the rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
