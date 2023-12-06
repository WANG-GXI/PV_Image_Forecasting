import csv
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras.preprocessing.image import array_to_img
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
predicted_images2:未经过自动编码器降噪处理的正确命名格式直接预测
predicted_images3:经过自动编码器处理的图片
predicted_images3_a:(visio)经过自动编码器处理的图片
predicted_images3_b:经过AE处理的正确命名格式的图片
predicted_images4:经过自动编码器处理并且预测
predicted_images4_a:经过自动编码器处理并且预测,预测和输入对比
predicted_images4_b:(visio)经过自动编码器处理并且预测,预测和实际对比
predicted_images4_c:经过自动编码器处理并且预测,预测和实际的二值化对比
predicted_images4_d:经过自动编码器处理并且预测,预测和实际的二值化对比,添加了评价参数和云量系数
predicted_images4_e:经过自动编码器处理正确命名格式的图片并且预测,预测和实际的二值化对比,添加了评价参数和云量系数
predicted_images5:(visio)经过变分自动编码器处理的图片
predicted_images5_a:(visio)经过AE-VAE处理的图片
'''
'''
1.暂时发现AE效果不如VAE
2.9-4：有一个新思路，现在是利用前5张图像预测后一张的，然后二值化得到云量--
---但是有个问题就是在修正辐照强度的时候前5个数据是空的，可以试试先提取出云量来，然后直接对数据进行预测
'''
# 图像文件夹路径
image_folder = r'E:\interpolated_images'
output_image_folder = r'E:\predicted_images4_word'  # 新的文件夹用于保存预测图像
csv_file_path  = r"D:\la432\batch_8e3844c8-60ee-45f5-8b8e-b3cee99a4402\csv_1.460806_103.788432_fixed_23_180_PT15M_Picture.xlsx"
def create_time_windows(images, window_size):
    """
    创建时间窗口数据集

    参数:
    images (numpy.ndarray): 包含图像数据的数组，形状为 (样本数, 高度, 宽度, 通道数)。
    window_size (int): 时间窗口的大小，即用于预测的历史图像数量。

    返回:
    X (numpy.ndarray): 时间窗口的输入数据，形状为 (样本数 - window_size, window_size, 高度, 宽度, 通道数)。
    y (numpy.ndarray): 时间窗口的输出数据，形状为 (样本数 - window_size, 高度, 宽度, 通道数)。
    """
    num_images = images.shape[0]
    height, width, channels = images.shape[1], images.shape[2], images.shape[3]
    X = []
    y = []
    for i in range(num_images - window_size):
        window = images[i:i + window_size]
        target = images[i + window_size]
        X.append(window)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return X, y
# 获取文件夹中所有图像文件的文件名
image_filenames = os.listdir(image_folder)

# 初始化空的列表来存储图像数据
images = []

# 遍历图像文件并加载它们
for filename in image_filenames:
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path, target_size=(64, 64), color_mode="grayscale")  # 加载灰度图像
        img_array = img_to_array(img) / 255.0  # 归一化像素值到 [0, 1] 范围
        # img_array = img_array[:, :, 0]  # 将图像从 (64, 64, 1) 转换为 (64, 64)，并仅保留一个通道（灰度图像）
        images.append(img_array)

# 将列表转换为NumPy数组
images = np.array(images)
# 设置时间步数
time_steps = 5
aaatestX, aaatestY = create_time_windows(images, time_steps)
# 划分训练集和测试集
# X_train, X_test = train_test_split(****, test_size=0.2, random_state=42)

# 构建循环神经网络 (RNN) 模型
model = keras.Sequential([
    layers.Input(shape=(5, 64, 64, 1)),  # 输入形状包括5张灰度图像
    # layers.Reshape((5, 64, 64, 1)),  # 将输入形状重塑为 (5, 64, 64, 1)
    layers.ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True),
    layers.ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=False),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(aaatestX, aaatestY, epochs=10, batch_size=32, validation_split=0.2)

# # 加载保存的模型
# model = tf.keras.models.load_model("pic_prediction_model.h5")
# 使用模型进行图像预测
predictions = model.predict(aaatestX)
#
# 遍历原始图像和预测结果，然后保存对比图像
# for i, (original_image, predicted_image) in enumerate(zip(aaatestY, predictions)):
#     # 反归一化到 [0, 255] 范围
#     original_image = original_image * 255.0
#     predicted_image = predicted_image * 255.0
#     predicted_img = array_to_img(predicted_image, scale=False)
#     # # 遍历5个输入图像和它们的预测结果
#     # if i >= 0:
#     #     for j in range(5):
#     #         # 创建图像对象
#     #         original_img = array_to_img(original_image[j], scale=False)
#     #         # 创建子图，将每个图像和预测结果放置在一个子图中
#     #         plt.subplot(2, 6, j + 1)  # 第一行显示输入图像
#     #         plt.title("Input Image{}".format(j + 1), fontsize=6, y=-0.25)
#     #         plt.imshow(original_img, cmap='gray')
#     #         plt.axis('off')
#     #         plt.subplot(2, 6, 6)  # 第二行显示预测结果
#     #         plt.title("Predicted Image", fontsize=6, y=-0.25)
#     #         plt.imshow(predicted_img, cmap='gray')
#     #         plt.axis('off')
#     #         # plt.subplot(2, 7, 7)  # 第二行显示预测结果
#     #         # plt.title("True Image", fontsize=6, y=-0.25)
#     #         # plt.imshow(array_to_img(original_image[-1], scale=False), cmap='gray')
#     #         # plt.axis('off')
#     #         # 调整子图之间的间距
#     #         plt.subplots_adjust(wspace=0.1)
#
#     # 创建图像对象
#     original_img = array_to_img(original_image, scale=False)
#     # 创建一个新的图像窗口
#     plt.figure(figsize=(5, 4))
#
#     # 显示原始图像
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image", fontsize=6, y=-0.1)
#     plt.imshow(original_img, cmap='gray')  # 使用 'gray' 颜色映射以适应灰度图像
#     plt.axis('off')  # 关闭坐标轴
#     plt.subplots_adjust(wspace=0.1)
#     # 显示预测的图像
#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Image", fontsize=6, y=-0.1)
#     plt.imshow(predicted_img, cmap='gray')  # 使用 'gray' 颜色映射以适应灰度图像
#     plt.axis('off')  # 关闭坐标轴
#     plt.subplots_adjust(wspace=0.1)
#     # 保存对比图像
#     output_image_path = os.path.join(output_image_folder, f'comparison_{i}.jpg')
#     plt.savefig(output_image_path, dpi=600, bbox_inches='tight')  # 保存整个图像，设置dpi和边界框以确保高分辨率和去除空白边界
#
#     # 关闭当前图像窗口以释放资源
#     plt.close()
# 遍历原始图像和预测结果，然后保存二值化对比图像，并且计算云的比例
# 创建一个空的DataFrame来存储CSV数据
df = pd.read_excel(csv_file_path)
# 初始化一个用于存储云的比例的空列表
cloud_percentages = []
cloud_percentages_original = []
for i, (original_image, predicted_image) in enumerate(zip(aaatestY, predictions)):
    # 计算MSE指标
    mse = np.mean((original_image - predicted_image) ** 2)
    # 计算PSNR
    psnr = cv2.PSNR(original_image, predicted_image)
    # 反归一化到 [0, 255] 范围
    original_image = original_image * 255.0
    predicted_image = predicted_image * 255.0

    # 二值化图像 阈值为127
    _, original_image_binary = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
    _, predicted_image_binary = cv2.threshold(predicted_image, 127, 255, cv2.THRESH_BINARY)
    # 计算原始结果中白色像素的数量（代表有云的像素）
    white_pixel_count_original = np.sum(original_image_binary == 255)
    total_pixel_count_original = original_image_binary.size
    # 计算预测结果中白色像素的数量（代表有云的像素）
    white_pixel_count = np.sum(predicted_image_binary == 255)
    total_pixel_count = predicted_image_binary.size
    # 计算云的比例
    original_cloud_percentages = (white_pixel_count_original / total_pixel_count_original) * 100
    cloud_percentage = (white_pixel_count / total_pixel_count) * 100

    cloud_percentages.append(cloud_percentage)
    cloud_percentages_original.append(original_cloud_percentages)
    # 创建一个新的图像窗口（plt.figure 不再需要）
    # 显示原始二值化图像
    plt.subplot(1, 3, 1)
    plt.title("True Image", fontsize=6, y=-0.1)
    plt.imshow(original_image_binary, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1)

    # 显示预测的二值化图像
    plt.subplot(1, 3, 2)
    plt.title("Predicted Image", fontsize=6, y=-0.1)
    plt.imshow(predicted_image_binary, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1)

    # 显示对比图像
    plt.subplot(1, 3, 3)
    plt.title("Comparison", fontsize=6, y=-0.25)
    plt.imshow(np.hstack([original_image_binary, predicted_image_binary]), cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1)
    # # 将MSE和PSNR指标以文字形式添加到子图中
    # plt.text(original_image.shape[1] - 150, original_image.shape[0] - 10, f'MSE: {mse:.4f}', fontsize=6, color='red')
    # plt.text(original_image.shape[1] - 150, original_image.shape[0] - 20, f'PSNR: {psnr:.2f}', fontsize=6, color='blue')
    # plt.text(original_image.shape[1] - 150, original_image.shape[0] - 30, f'original Cloud Percentage: {original_cloud_percentages:.2f}%',
    #          fontsize=6, color='green')
    # plt.text(original_image.shape[1] - 150, original_image.shape[0] - 40, f'Cloud Percentage: {cloud_percentage:.2f}%',
    #          fontsize=6, color='green')

    # 保存对比图像
    output_image_path = os.path.join(output_image_folder, f'comparison_binary_{i}.jpg')
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')

    # 关闭当前图像窗口以释放资源
    plt.close()
# 在cloud_percentages前面添加5个来自原始云图的数据
# cloud_percentages = cloud_percentages_original[:5] + cloud_percentages
# 进行线性插值以填充前五个数据
x = np.arange(0, len(cloud_percentages))
f = interp1d(x, cloud_percentages, kind='linear', fill_value="extrapolate")
# 根据插值函数计算前五个数据
additional_cloud_data = f(np.arange(-5, 0))
# 将新增数据追加到cloud_percentages中
cloud_percentages = list(additional_cloud_data) + cloud_percentages
# 将插值后的云层系数替换掉前五个数据
df['Cloud Percentage_novae'].iloc[:5] = additional_cloud_data
# 如果数据行数不足，将添加默认值 -1
df['Cloud Percentage_novae'] = cloud_percentages+[-1] * (len(df) - len(cloud_percentages))

# 保存更新后的DataFrame到CSV文件
df.to_excel(csv_file_path, index=False)
# 显示完成
print("图像保存完成。")
