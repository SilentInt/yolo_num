from ultralytics import YOLO
import cv2
import numpy as np


def pre_process(image):

    # 将图片转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黑色的阈值范围
    lower_black = np.array([0, 0, 0])
    # upper_black = np.array([180, 255, 50])
    upper_black = np.array([180, 255, 60])

    # 根据阈值创建掩膜
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 对掩模进行膨胀操作
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)
    return mask


# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model

img = cv2.imread('1.jpg')
# 预处理
img = pre_process(img)

# 将二值图像转换为灰度图像
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# Run YOLOv8 inference on the frame
results = model(img)[0]
print(model)
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
