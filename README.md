# Object-detection-using-web-camera

# YOLOv4-Real-Time-Object-Detection

YOLOv4 Real-Time Object Detection is an exciting project that uses deep learning and OpenCV to detect objects live through your webcam.

Welcome to the YOLOv4 Real-Time Object Detection project — a simple yet powerful implementation that shows how computers can identify objects around you instantly! This repository demonstrates how to combine OpenCV, a pre-trained YOLOv4 model, and your system’s camera to visualize object detection in real time. Whether you're exploring AI, learning computer vision, or just experimenting with cool technology, this project is perfect for you!

# Features:
1.Detects 80+ everyday objects (person, car, bottle, dog, etc.)

2.Uses YOLOv4, a fast and accurate real-time object detection model

3.Displays bounding boxes with confidence scores

4.Works directly with your webcam

5.Smooth and real-time output

# Technologies Used:
1.Python

2.OpenCV for camera access and image processing

3.NumPy for efficient numerical operations

4.YOLOv4 (weights + config + COCO class labels)

# How to Use:
1.Clone this repository.

2.Download the YOLOv4 model files (yolov4.weights, yolov4.cfg, coco.names).

3.Place all the files in your project folder.

4.Run the Python script to start real-time detection.

5.Press 'q' anytime to quit the webcam window.

# Applications:
1.Learning how deep learning models perform object detection

2.Understanding image processing fundamentals

3.Real-time AI projects for beginners and students

4.Experimenting with computer vision workflows

5.Robotics, surveillance demos, smart camera prototypes

# Program:
```

import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

# Save image counter
img_counter = 0

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Failed to grab frame")
        continue

    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
 ```

# Output:

<img width="1920" height="1200" alt="Screenshot 2025-11-15 134301" src="https://github.com/user-attachments/assets/cf84a338-c6d9-4c75-95f6-5704964d1764" />
