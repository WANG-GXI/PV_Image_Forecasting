# 项目名称: 图像时间分辨率15分钟线性插值
# 版本号: 1.0
# 作者: [王昱栋]
# 日期: [2023-9-4]
import os
import re
import pandas as pd
import numpy as np
from PIL import Image

# 设置原始图像文件夹和目标文件夹
source_folder = r'E:\handle_pic_2_stander'
output_folder = r'E:\interpolated_images'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# 创建一个Pandas DataFrame来存储图像的时间信息
image_data = {'file_name': [], 'timestamp': []}

# 从图像文件名中提取时间戳信息
for image_file in image_files:
    # 使用正则表达式提取时间戳部分
    match = re.search(r'NC_H08_(\d{8}_\d{4})_R21_FLDK_\d+.jpg', image_file)
    if match:
        timestamp_str = match.group(1)
        # 将时间戳字符串转换为日期时间对象，并增加8小时
        timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M') + pd.Timedelta(hours=8)
        image_data['file_name'].append(image_file)
        image_data['timestamp'].append(timestamp)

# 将数据转换为Pandas DataFrame
df = pd.DataFrame(image_data)

# 创建一个新的时间序列，以15分钟为间隔
start_time = df['timestamp'].min()
end_time = df['timestamp'].max()
new_time_range = pd.date_range(start=start_time, end=end_time, freq='15T')

# 使用线性插值生成新的时间戳点对应的图像
for i in range(len(new_time_range) - 1):
    start_timestamp = new_time_range[i]
    end_timestamp = new_time_range[i + 1]

    # 找到最接近的两个已有时间戳
    closest_start_idx = df['timestamp'].sub(start_timestamp).abs().idxmin()
    closest_end_idx = df['timestamp'].sub(end_timestamp).abs().idxmin()

    # 获取对应的文件名
    start_image_file = df.loc[closest_start_idx, 'file_name']
    end_image_file = df.loc[closest_end_idx, 'file_name']

    # 执行线性插值
    start_image = Image.open(os.path.join(source_folder, start_image_file))
    end_image = Image.open(os.path.join(source_folder, end_image_file))

    # 计算权重
    weight_end = (end_timestamp - start_timestamp) / (df.loc[closest_end_idx, 'timestamp'] - df.loc[closest_start_idx, 'timestamp'])
    weight_start = 1 - weight_end

    # 执行线性插值
    interpolated_image = Image.blend(start_image, end_image, weight_end)

    # 保存插值后的图像到目标文件夹，注意修改时间戳
    interpolated_timestamp_str = start_timestamp.strftime('%Y%m%d_%H%M')
    interpolated_image_file = f'NC_H08_{interpolated_timestamp_str}_R21_FLDK.jpg'
    interpolated_image.save(os.path.join(output_folder, interpolated_image_file))

print("插值完成")
