import numpy as np
import netCDF4 as nc
from osgeo import gdal, osr, ogr
import glob


def nc2tif(data, Output_folder,i):
    pre_data = nc.Dataset(data)  # 利用.Dataset()读取nc数据
    Lat_data = pre_data.variables['latitude'][:]
    Lon_data = pre_data.variables['longitude'][:]
    print(pre_data.variables.keys())
    pre_arr= np.asarray(pre_data.variables['albedo_01'])#属性变量名
    # 获取所有波段的数据
    bands = ['albedo_01', 'albedo_02','albedo_03','albedo_04','albedo_05','tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16']
    # bands = ['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16']

    band_data = [np.asarray(pre_data.variables[band]) for band in bands]
    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    # Lonmin, Latmax, Lonmax, Latmin
    # 分辨率计算
    Num_lat = len(Lat_data)
    Num_lon = len(Lon_data)
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = Output_folder + '\\' + data.split('\\')[-1].split('.')[0] + '_' + str(i + 1) + '.tif'
    out_tif = driver.Create(out_tif_name, Num_lon, Num_lat, len(band_data), gdal.GDT_Float32)
    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    out_tif.SetGeoTransform(geotransform)
    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    # 数据导出
    for j, band in enumerate(band_data):
        out_tif.GetRasterBand(j + 1).WriteArray(band)  # 将每个波段的数据写入相应的波段
    # out_tif.GetRasterBand(1).WriteArray(pre_arr)  # 将数据写入内存
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件

if __name__ == "__main__":
    # Download_Path用于存储下载的原始数据
    Download_Path = r"E:\nc_data"
    # Analysis_Path用于存储处理后的数据（即转为TIFF后的数据）的文件夹
    Analysis_Path = r"E:\tif_data"
    # 下面开始数据处理
    # 读取所有nc数据
    data_list = glob.glob(Download_Path + "\\*.nc")
    # for循环完成解析
    for i in range(len(data_list)):
        data = data_list[i]
        nc2tif(data, Analysis_Path,i)
        # NC_to_tiffs(data, Analysis_Path)
        print(data + "-----转tif成功")
    print("----转换结束----")
