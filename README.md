# PV_Image_Forecasting
# 使用Pycharm-Tensorflow
## 海上光伏发电+图像云层预测
## 运行流程：
          【1】data.py：下载云图数据，格式为.nc
          【2】nc_data.py：叠加全波段数据，格式从.nc到.tiff
          【3】handle_tif.py：去冗余+加权高斯滤波，格式从.tiff到.jpg
          【4】pic_stander.py：直方图均衡化，格式.jpg
          【5】pic_插值.py：时间分辨率从10分钟到15分钟，格式.jpg
          【6】handle_pic.py：对图像使用编码器降噪，格式.jpg
          【7】handle2.py：前5张图像预测第六张的结果
          【8】handle3.py：完整版的图像预测+输出结果保存（包含云层系数的计算）
          
