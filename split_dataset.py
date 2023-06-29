import os
import random
import shutil

# 设置原始数据集路径和目标路径
dataset_path = "gen"
train_images_path = "datasets/images/train"
train_labels_path = "datasets/labels/train"

val_images_path = "datasets/images/val"
val_labels_path = "datasets/labels/val"
test_images_path = "datasets/test_images"

# 创建目标文件夹
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

# 获取原始图像文件列表
images_folder = os.path.join(dataset_path, "images")
image_files = [f for f in os.listdir(
    images_folder) if os.path.isfile(os.path.join(images_folder, f))]

# 随机打乱图像文件列表
random.shuffle(image_files)

# 确定测试集的图像数量（这里假设使用10%的图像作为测试集）
num_test_images = int(0.1 * len(image_files))

# 打印图像数量
print("Total images: ", len(image_files))

print("Num test images: ", num_test_images)
print("Num train images: ", len(image_files) - num_test_images)

# 复制测试集的图像到测试集的文件夹中
for i in range(num_test_images):
    image_file = image_files[i]
    image_src = os.path.join(images_folder, image_file)
    image_dst = os.path.join(val_images_path, image_file)
    shutil.copy(image_src, image_dst)

    # 对应的标签文件
    label_file = image_file.replace(".jpg", ".txt")
    label_src = os.path.join(dataset_path, "labels", label_file)
    label_dst = os.path.join(val_labels_path, label_file)
    shutil.copy(label_src, label_dst)

# 复制剩余的图像到训练集的文件夹中
for i in range(num_test_images, len(image_files)):
    image_file = image_files[i]
    image_src = os.path.join(images_folder, image_file)
    image_dst = os.path.join(train_images_path, image_file)
    shutil.copy(image_src, image_dst)

    # 对应的标签文件
    label_file = image_file.replace(".jpg", ".txt")
    label_src = os.path.join(dataset_path, "labels", label_file)
    label_dst = os.path.join(train_labels_path, label_file)
    shutil.copy(label_src, label_dst)
