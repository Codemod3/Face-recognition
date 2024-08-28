import cv2
import face_recognition

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
        cp = frame[top:bottom, left:right]

        # Calculated with numpy.mean()
        age_mean = (78.4263377603, 87.7689143744, 114.895847746)

        # Preprocess the face region for prediction
        face_blob = cv2.dnn.blobFromImage(cp, 1, (227, 227), age_mean, swapRB=True)

        # Gender prediction
        gender_label = ['Male', 'Female']
        gender_protext = 'dataset/gender_deploy.prototxt'
        gender_caffemodel = 'dataset/gender_net.caffemodel'
        gender_neural_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_neural_net.setInput(face_blob)
        gender_prediction = gender_neural_net.forward()
        gender = gender_label[gender_prediction[0].argmax()]

        # Age prediction
        age_label = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        age_protext = 'dataset/age_deploy.prototxt'
        age_caffemodel = 'dataset/age_net.caffemodel'
        age_neural_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_neural_net.setInput(face_blob)
        age_prediction = age_neural_net.forward()
        age = age_label[age_prediction[0].argmax()]

        # Display the gender and age on the video
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, gender + " " + age, (left, bottom + 20), font, 0.5, (255, 255, 255), 1)

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
