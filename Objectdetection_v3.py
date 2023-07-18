import cv2
import numpy as np

# Load the pre-trained SSD model
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

# List of COCO class labels
coco_classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"]

def detect_objects(image):
    # Convert image to blob format
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run forward pass through the network
    detections = net.forward()

    # Loop over the detections and draw boxes around detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold for detection
            class_id = int(detections[0, 0, i, 1])
            class_name = coco_classes[class_id]
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Access the webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam, change if you have multiple webcams

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output_frame = detect_objects(frame)

    cv2.imshow("Object Detection", output_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
