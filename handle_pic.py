# 项目名称: 自动编码器降噪
# 版本号: 1.0
# 作者: [王昱栋]
# !2023-9-4发现：对图像进行15分钟时间分辨率插值后预测效果下降，发现原因是使用了新的模型，换为原先10分钟分辨率的模型预测效果恢复，暂时认为是样本数量严重减小导致的
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
# 图像文件夹路径
image_folder = r'E:\handle_pic_2_stander'
output_image_folder = r'E:\predicted_images3'  # 新的文件夹用于保存预测图像
# 获取文件夹中所有图像文件的文件名
image_filenames = os.listdir(image_folder)

# 初始化空的列表来存储图像数据
images = []

# 遍历图像文件并加载它们
for filename in image_filenames:
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path, target_size=(64, 64), color_mode="grayscale")  # 调整目标大小
        img_array = img_to_array(img) / 255.0  # 归一化像素值到 [0, 1] 范围
        images.append(img_array)

# 将列表转换为NumPy数组
X = np.array(images)

# 划分训练集和测试集
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# # 构建自动编码器模型,注意：下面代码只运行一次，训练完成之后保存h5模型即可直接调用
# encoder_input = keras.Input(shape=(64, 64, 1))
# x = layers.Flatten()(encoder_input)
# encoded = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(64, activation='relu')(encoded)
# decoded = layers.Dense(64 * 64 * 1, activation='sigmoid')(x)
# decoded = layers.Reshape((64, 64, 1))(decoded)
#
# autoencoder = keras.Model(encoder_input, decoded)
#
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#
# # 训练自动编码器
# autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2)

autoencoder = tf.keras.models.load_model("my_autoencoder_model.h5")
# 使用自动编码器进行图像预测
predictions = autoencoder.predict(X)
# 只保存预测结果
for i, predicted_image in enumerate(predictions):
    # 反归一化到 [0, 255] 范围
    predicted_image = predicted_image * 255.0

    # 创建图像对象
    predicted_img = array_to_img(predicted_image, scale=False)

    # 创建一个新的图像窗口
    plt.figure(figsize=(5, 5))

    # 显示预测的图像
    plt.imshow(predicted_img, cmap='gray')  # 使用 'gray' 颜色映射以适应灰度图像
    plt.axis('off')  # 关闭坐标轴

    # 保存预测图像
    output_image_path = os.path.join(output_image_folder, f'prediction_{i}.jpg')
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight', pad_inches=0.0)  # 保存整个图像，设置dpi和边界框以确保高分辨率和去除空白边界

    # 关闭当前图像窗口以释放资源
    plt.close()
# 保存自动编码器模型
# autoencoder.save('my_autoencoder_model.h5')

# # 保存对比图像
# for i, (original_image, predicted_image) in enumerate(zip(X, predictions)):
#     # 计算MSE指标
#     mse = np.mean((original_image - predicted_image) ** 2)
#     # 计算PSNR
#     psnr = cv2.PSNR(original_image, predicted_image)
#     # 反归一化到 [0, 255] 范围
#     original_image = original_image * 255.0
#     predicted_image = predicted_image * 255.0
#
#     # 创建图像对象
#     original_img = array_to_img(original_image, scale=False)
#     predicted_img = array_to_img(predicted_image, scale=False)
#
#     # 创建一个新的图像窗口
#     plt.figure(figsize=(10, 9))
#
#     # 显示原始图像
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image", fontsize=7.5)
#     plt.imshow(original_img, cmap='gray')  # 使用 'gray' 颜色映射以适应灰度图像
#     plt.axis('off')  # 关闭坐标轴
#
#     # 显示预测的图像
#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Image", fontsize=7.5)
#     plt.imshow(predicted_img, cmap='gray')  # 使用 'gray' 颜色映射以适应灰度图像
#     plt.axis('off')  # 关闭坐标轴
#     # 将MSE和PSNR指标以文字形式添加到子图中
#     plt.text(original_image.shape[1] - 150, original_image.shape[0] - 10, f'MSE: {mse:.4f}', fontsize=7, color='red')
#     plt.text(original_image.shape[1] - 150, original_image.shape[0] - 20, f'PSNR: {psnr:.2f}', fontsize=7, color='blue')
#     # 保存对比图像
#     output_image_path = os.path.join(output_image_folder, f'comparison_{i}.jpg')
#     plt.savefig(output_image_path, dpi=600, bbox_inches='tight')  # 保存整个图像，设置dpi和边界框以确保高分辨率和去除空白边界
#
#     # 关闭当前图像窗口以释放资源
#     plt.close()

# 显示完成
print("图像保存完成。")
