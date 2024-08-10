import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
        frame[y:y+h, x:x+w] = blurred_roi
    cv2.imshow('Webcam Feed - Press "q" to exit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
