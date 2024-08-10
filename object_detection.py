import cv2
import numpy as np
import requests

# URLs
'''urls = {
    'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
    'yolov3.cfg': 'https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg',
    'coco.names': 'https://github.com/pjreddie/darknet/raw/master/data/coco.names'
}

# Download each file
for filename, url in urls.items():
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")'''

net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 0, 255)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("YOLO Object Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
