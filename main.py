from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

img = '1.jpg'

# Run YOLOv8 inference on the frame
results = model(img)[0]

# Visualize the results on the frame
annotated_frame = results.plot()

# print(int(results[0].boxes.cpu().cls[0]))
boxes = results.boxes.cpu()
print(boxes)

# 将所有识别出的物体的横坐标和类别存入列表
box_list = []
for i in range(len(boxes)):
    box_list.append((boxes.xyxy[i][0], boxes.cls[i]))

# Sort objects by x-coordinate
box_list_sorted = sorted(box_list, key=lambda x: x[0])
print(box_list_sorted)

# 按横坐标顺序输出所有识别出的物体
for box in box_list_sorted:
    print(results.names[int(box[1])])

cv2.imshow("YOLOv8n", annotated_frame)

# Press q to quit
while True:
    if cv2.waitKey(1) == ord('q'):
        break
