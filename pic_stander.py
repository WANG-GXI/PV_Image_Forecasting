# 项目名称: 直方图均衡化
# 版本号: 1.0
# 作者: [王昱栋]
import cv2
import os
from PIL import Image

# 输入文件夹路径和输出文件夹路径
# !不能在路径和图片名字中出现中文
input_folder = r"E:\MSEpic"  # 请替换为你的输入文件夹路径
output_folder = r"E:\MSE_pic1" # 请替换为你的输出文件夹路径

# 确保输出文件夹存在，如果不存在则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# 循环处理每张图像
for image_file in image_files:
    # 构建图像的完整路径
    image_path = os.path.join(input_folder, image_file)

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从 BGR 格式转换为灰度格式
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)

    # 将均衡化后的灰度图像转换回 BGR 格式
    equalized_bgr_image = cv2.cvtColor(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

    # 构建输出图像的完整路径
    output_path = os.path.join(output_folder, image_file)

    # 保存均衡化后的图像，设置 DPI 为 600
    pil_image = Image.fromarray(equalized_bgr_image)
    pil_image.save(output_path, dpi=(600, 600))

    print("处理完成：", image_path, "保存为：", output_path)

print("批量处理完成。")

