import cv2
import numpy as np
import os
import random
import shutil
from PIL import Image, ImageDraw, ImageFont

# 指定字体文件路径
font_path = "./SourceHanSansSC-Bold.ttf"

output_dir = "./images/"  # 图片输出目录
label_dir = "./labels/"  # 标签输出目录
# edges_dir = "./edges/"  # 边缘输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
# os.makedirs(edges_dir, exist_ok=True)


def generate_images(images_amount):
    """生成指定数量的图片和标签"""

    image_size = (800, 600)

    for i in range(0, images_amount):
        # 使用Pillow库加载自定义字体
        font_size = random.randint(50, 150)
        font = ImageFont.truetype(font_path, font_size)

        digit = str(random.randint(0, 9))
        img = np.zeros(image_size, dtype=np.uint8)

        # 创建Pillow图像对象并绘制数字
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # 计算数字的位置
        digit_width, digit_height = draw.textlength(
            digit, font=font), font_size
        position = ((image_size[0] - digit_width) // 2,
                    (image_size[1] - digit_height) // 2)
        draw.text(position, digit, font=font, fill=255)

        # 将Pillow图像对象转换回NumPy数组
        img = np.array(pil_img)

        # 随机生成平移、旋转和透视变换参数
        tx = random.randint(-300, 300)
        ty = random.randint(-100, 100)
        angle = random.randint(-50, 50)
        perspective = np.float32([[1, 0, random.uniform(-0.001, 0.001)],
                                  [0, 1, random.uniform(-0.001, 0.001)],
                                  [0, 0, 1]])

        # 应用平移、旋转和透视变换
        img = cv2.warpAffine(img, np.float32(
            [[1, 0, tx], [0, 1, ty]]), image_size)
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D(
            (image_size[0] // 2, image_size[1] // 2), angle, 1), image_size)
        img = cv2.warpPerspective(img, perspective, image_size)

        # 反转颜色，变为白底黑字
        img = cv2.bitwise_not(img)

        # 使用边缘检测函数查找数字的边缘
        edges = cv2.Canny(img, 50, 255)

        # 进行闭运算操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 保存边缘图像
        # cv2.imwrite(f"{edges_dir}{i}.jpg", edges)

        # 根据边缘图像获取轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(max_contour)

        # 计算数字的中心坐标、宽度和高度
        center_x = x + w // 2
        center_y = y + h // 2
        width = w
        height = h

        # 归一化坐标
        center_x_norm = center_x / image_size[0]
        center_y_norm = center_y / image_size[1]
        width_norm = width / image_size[0]
        height_norm = height / image_size[1]

        # 保存标签
        label = f"{digit} {center_x_norm} {center_y_norm} {width_norm} {height_norm}"

        # 保存图片
        cv2.imwrite(f"{output_dir}{i}.jpg", img)

        # 保存标签到txt文件
        with open(f"{label_dir}{i}.txt", "w") as file:
            file.write(f"{label}\n")


generate_images(3000)
shutil.copy('./classes.txt', label_dir)
