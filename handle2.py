import os
from PIL import Image
import numpy as np
from keras.layers import Conv2D, ConvLSTM2D, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
#以下代码只需要训练一次，保存h5模型后即可实现调用
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # 设置图像文件夹路径
# image_folder = 'E:/handle_pic_2_stander'
#
# # 获取图像文件列表
# image_files = os.listdir(image_folder)
# image_files.sort()  # 确保图像按顺序排列
# height=64
# width =64
# # 定义一些参数
# num_frames = 5  # 每个序列包含的帧数
# image_shape = (height, width)  # 图像的尺寸，根据您的图像实际尺寸设置
# channels=3
# # 创建一个空的数组来存储图像序列
# image_sequences = []
#
# # 循环遍历图像文件并创建序列
# for i in range(num_frames, len(image_files)):
#     image_sequence = []
#     for j in range(num_frames):
#         # 读取图像并将其调整为所需的尺寸
#         image = Image.open(os.path.join(image_folder, image_files[i - num_frames + j]))
#         image = image.resize(image_shape)
#         image = np.array(image)
#         # 如果图像是彩色的，您可能需要将其转换为灰度图像
#         # image = np.mean(image, axis=2)
#         image_sequence.append(image)
#     image_sequences.append(image_sequence)
#
# # 将图像序列转换为NumPy数组
# image_sequences = np.array(image_sequences)
#
# # 现在，image_sequences 包含了每个序列的图像数据
# # 您可以继续使用前面提供的ConvLSTM模型代码来训练和测试模型
#
# # 划分训练集和测试集
# train_ratio = 0.8
# num_samples = image_sequences.shape[0]
# num_train_samples = int(train_ratio * num_samples)
# x_train = image_sequences[:num_train_samples]
# x_test = image_sequences[num_train_samples:]
# # 构建ConvLSTM模型
# model = Sequential()
# model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(5, height, width, channels), padding='same', return_sequences=True))
# model.add(BatchNormalization())
# model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
# model.add(BatchNormalization())
# model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
# model.add(BatchNormalization())
# model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
#
# # 编译模型
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # 训练模型
# model.fit(x_train, x_train, epochs=10, batch_size=32, validation_data=(x_test, x_test))
# model.save('my_conv_lstm_model.h5')
# sample_index = 0
# input_sequence = x_test[sample_index:sample_index+1]
#
# # 使用模型生成预测
# predicted_frame = model.predict(input_sequence)

# 加载保存的模型

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 设置图像文件夹路径
image_folder = r'E:/handle_pic_2_stander'

# 获取图像文件列表
image_files = os.listdir(image_folder)
image_files.sort()  # 确保图像按顺序排列
height=64
width =64
# 定义一些参数
num_frames = 5  # 每个序列包含的帧数
image_shape = (height, width)  # 图像的尺寸
channels=3
# 创建一个空的数组来存储图像序列
image_sequences = []

# 循环遍历图像文件并创建序列
for i in range(num_frames, len(image_files)):
    image_sequence = []
    for j in range(num_frames):
        # 读取图像并将其调整为所需的尺寸
        image = Image.open(os.path.join(image_folder, image_files[i - num_frames + j]))
        image = image.resize(image_shape)
        image = np.array(image)
        # 如果图像是彩色的，将其转换为灰度图像
        # image = np.mean(image, axis=2)
        image_sequence.append(image)
    image_sequences.append(image_sequence)

# 将图像序列转换为NumPy数组
input_sequence = np.array(image_sequences)
# 加载保存的模型
loaded_model = load_model('my_conv_lstm_model.h5')

# 使用加载的模型进行预测
predicted_sequence = loaded_model.predict(input_sequence)

# 灰度图像将输出数据缩放到合适的范围（ 0 到 1 之间）
# 然后将其转换为图像并保存
output_folder = 'E:/predicted_images3'
os.makedirs(output_folder, exist_ok=True)

for i, predicted_frame in enumerate(predicted_sequence):
    # 将输出数据缩放到合适的范围（0到255之间）
    predicted_frame = (predicted_frame * 255).astype(np.uint8)
    # predicted_frame = predicted_frame.reshape((64, 64))
    # 创建预测结果的图像
    predicted_image = Image.fromarray(predicted_frame)

    # 保存预测结果为图像文件
    predicted_image.save(os.path.join(output_folder, f'predicted_image_{i}.jpg'))
