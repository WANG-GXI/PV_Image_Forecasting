import os
import rasterio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
'''
handle_pic： 高斯滤波 参数为1       有些模糊
handle_pic_1：高斯滤波 参数为0.001   模糊减弱
handle_pic_2：无高斯滤波全圆盘
'''
# 源文件夹和目标文件夹的路径
input_folder = r"E:\tif_data"
output_folder = r"E:\handle_pic_visio4"

# 目标经纬度
target_lat = 1 + 48 / 60 + 9.08 / 3600  # 转换为度
target_lon = 103 + 58 / 60 + 58.76 / 3600  # 转换为度
squared_diff,mi,ssim_value,mse,psnr_test= 0,0,0,0,0

# 遍历输入文件夹中的所有tif文件
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        tif_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

        # 使用rasterio打开TIFF数据
        with rasterio.open(tif_path) as dataset:
            # 获取地理转换参数
            transform = dataset.transform

            # 将目标经纬度转换为像素坐标
            col, row = dataset.index(target_lon, target_lat)

            # 计算矩形像素坐标范围
            rect_width = 200  # 例如，以目标点为中心，取200个像素的宽度
            rect_height = 200  # 例如，取200个像素的高度
            rect_x = col - rect_width // 2
            rect_y = row - rect_height // 2

            # 读取矩形区域的图像数据
            image_data = dataset.read(window=((rect_x, rect_x + rect_width), (rect_y, rect_y + rect_height)))
            # # 融合多波段图像为单波段图像
            # merged_image = image_data.mean(axis=0)  # 使用均值融合
            # # 使用导向滤波
            # guided_image = cv2.ximgproc.guidedFilter(merged_image.astype(np.float32),
            #                                          merged_image.astype(np.float32), radius=5, eps=10)
            # 定义不同波段的权重，你可以根据需要进行调整
            weight_albedo = 0.297  # 例如，亮度的权重为0.2
            weight_albedo1 = 0.792  # 例如，亮度的权重为0.2
            weight_tbb = 0.001  # 其他波段的总权重为0.8

            # 应用高斯滤波器到每个波段
            blurred_bands = [gaussian_filter(band, sigma=0.001) for band in image_data]

            # # 将多波段数据按权重合并为单通道灰度图像
            merged_image = (
                     weight_albedo * np.mean(blurred_bands[:3], axis=0) +  # 前3个波段使用亮度的权重
                     weight_albedo1 * np.mean(blurred_bands[3:5], axis=0) +  # 中间3个波段使用亮度的权重
                     weight_tbb * np.mean(blurred_bands[5:], axis=0)  # 其余波段使用其他波段的总权重
            )
            
            # 使用Matplotlib显示图像
            plt.figure(figsize=(10, 9))
            plt.imshow(merged_image, cmap='gray')

            # 关闭坐标轴
            plt.axis('off')

            # 保存图像为PNG文件
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=600)

            # 关闭Matplotlib图形
            plt.close()
            # # 评价融合指标(论文中需要展示结果时取消注释)
            # for i in range(len(image_data)):
            #     psnr_test += compare_psnr(image_data[i].astype("float"),merged_image,data_range=merged_image.max() - merged_image.min())
            #     squared_diff = (image_data[0].astype("float") - merged_image) ** 2
            #     mi += mutual_info_score(image_data[i].ravel(), merged_image.ravel())
            #     ssim_value += ssim(image_data[i], merged_image, data_range=merged_image.max() - merged_image.min())
            #     mse = squared_diff.mean()
            # print(f"MSE between original and processed image: {mse}")
            # # 计算互信息(Mutual Information, MI)
            # print(f"Mutual Information (MI) between original and processed image: {mi/15}")
            # # 计算结构相似性指数度量(SSIM)
            # print(f"Structural Similarity Index (SSIM) between original and processed image: {ssim_value/15}")
            # # 计算结构相似性指数度量(SSIM)
            # print(f"TEST Structural Similarity Index (SSIM) between original and processed image: {psnr_test/15}")
print("Conversion and cropping completed.")


