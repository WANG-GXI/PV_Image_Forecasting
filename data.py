#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ================================================================================================
# Author: Jianpu | Affiliation: Hohai
# Email : 211311040008@hhu.edu.cn
# Last modified: 2023-04-04 12:28:06
# Filename: download_kuihua8_L1.py
# Description: 1.日本葵花8号卫星，L1级数据产品下载脚本
#              2.需要提前下载好对于葵花8卫星产品下载的安装包： lb_toolkits.tools
#              3.在官网注册获取下载的用户名和密码以及链接网址
#              4. 可以选择两个分辨率进行下载：5km和2km
#			   5. 可以选择下载的频率：10min一次或者30min一次
# =================================================================================================

"""
import os
import sys
import datetime
import time
from lb_toolkits.tools import ftppro


class downloadH8(object):

    def __init__(self, username, password):

        self.ftp = ftppro(FTPHOST, username, password)

    def search_ahi8_l1_netcdf(self, starttime, endtime=None, pattern=None, skip=False):
        '''
        下载葵花8号卫星L1 NetCDF数据文件
        Parameters
        ----------
        starttime : datetime
            下载所需数据的起始时间
        endtime : datetime
            下载所需数据的起始时间
        pattern: list, optional
            模糊匹配参数
        Returns
        -------
            list
            下载的文件列表
        '''

        if endtime is None:
            endtime = starttime

        downfilelist = []

        nowdate = starttime
        while nowdate <= endtime:
            # 拼接H8 ftp 目录
            sourceRoot = os.path.join('/jma/netcdf', nowdate.strftime("%Y%m"), nowdate.strftime("%d"))
            sourceRoot = sourceRoot.replace('\\', '/')

            # 获取文件列表
            filelist = self.GetFileList(starttime, endtime, sourceRoot, pattern)

            # filelist = [f for f in filelist if f.startswith('NC_H08_') and f.endswith('.06001_06001.nc')]

            if len(filelist) == 0:
                nowdate += datetime.timedelta(days=1)
                print('未匹配当前时间【%s】的文件' % (nowdate.strftime('%Y-%m-%d')))
                continue

            nowdate += datetime.timedelta(days=1)
            downfilelist.extend(filelist)

        return downfilelist

    def GetFileList(self, starttime, endtime, srcpath, pattern=None):
        ''' 根据输入时间，匹配获取H8 L1数据文件名  '''
        downfiles = []

        srcpath = srcpath.replace('\\', '/')

        filelist = self.ftp.listdir(srcpath)
        filelist.sort()
        for filename in filelist:
            namelist = filename.split('_')
            nowdate = datetime.datetime.strptime('%s %s' % (namelist[2], namelist[3]), '%Y%m%d %H%M')

            if (nowdate < starttime) | (nowdate > endtime):
                continue

            downflag = True
            # 根据传入的匹配参数，匹配文件名中是否包含相应的字符串
            if pattern is not None:
                if isinstance(pattern, list):
                    for item in pattern:
                        if item in filename:
                            downflag = True
                            # break
                        else:
                            downflag = False
                            break
                elif isinstance(pattern, str):
                    if pattern in filename:
                        downflag = True
                    else:
                        downflag = False

            if downflag:
                srcname = os.path.join(srcpath, filename)
                srcname = srcname.replace('\\', '/')

                downfiles.append(srcname)

        return downfiles

    def download(self, outdir, srcfile, blocksize=2048, skip=False):
        """通过ftp接口下载H8 L1数据文件"""

        if not os.path.exists(outdir):
            os.makedirs(outdir)
            print('成功创建路径：%s' % (outdir))

        if isinstance(srcfile, list):
            count = len(srcfile)
            for srcname in srcfile:
                count -= 1
                self._download(outdir, srcname, blocksize=blocksize, skip=skip, count=count + 1)

        elif isinstance(srcfile, str):
            self._download(outdir, srcfile, blocksize=blocksize, skip=skip)

    def _download(self, outdir, srcname, blocksize=2048, skip=False, count=1):

        print('=' * 100)
        basename = os.path.basename(srcname)
        dstname = os.path.join(outdir, basename)

        if skip:
            return srcname

        if os.path.isfile(dstname):
            print('文件已存在，跳过下载>>【%s】' % (dstname))
            return srcname

        stime = time.time()
        print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
              '开始下载文件【%d】: %s' % (count, srcname))

        if self.ftp.downloadFile(srcname, outdir, blocksize=blocksize):
            print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                  '成功下载文件【%s】:%s' % (count, dstname))
        else:
            print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                  '下载文件失败【%s】:%s' % (count, dstname))

        etime = time.time()
        print('下载文件共用%.2f秒' % (etime - stime))

        return srcname


def check_data_completeness(file_list, start_time, end_time):
    expected_num_files = (end_time - start_time).days * 144 + 144  # 48 show 30min/time; 144 show 10min/time
    actual_num_files = len(file_list)

    if actual_num_files == expected_num_files:
        print("已经下载了全部数据。")
    else:
        print("有 %d 个数据文件缺失。" % (expected_num_files - actual_num_files))
        expected_file_names = []
        actual_file_names = []

        for i in range(expected_num_files):

            file_time = start_time + datetime.timedelta(minutes=i * 10)
            file_name = "NC_H08_%s_R21_FLDK.06001_06001.nc" % (file_time.strftime("%Y%m%d_%H%M"))
            expected_file_names.append(file_name)

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            actual_file_names.append(file_name)

        missing_file_names = set(expected_file_names) - set(actual_file_names)

        for missing_file_name in missing_file_names:
            print("缺失文件：%s" % missing_file_name)


FTPHOST = 'ftp.ptree.jaxa.jp'

# create an instance of the downloadH8 class
h8_downloader = downloadH8('2371016_stu.neu.edu.cn', 'SP+wari8')
#
# search for H8 files for a specific date
start_time = datetime.datetime(2021, 1, 2)
end_time = datetime.datetime(2021, 2, 2)
file_list = h8_downloader.search_ahi8_l1_netcdf(start_time, end_time, pattern=['R21', '06001_06001'])


# 打印选取的文件名
print(file_list)

check_data_completeness(file_list, start_time, end_time)

from tqdm import tqdm

for file in tqdm(file_list):
    if (int(file[38:40]) >= 22 or int(file[38:40]) <= 10):
        try:
            h8_downloader.download(r'E:\nc_data', file)
        except ValueError as e:
            print(str(e))
            os.remove(os.path.join(r'E:\nc_data', os.path.basename(file)))
            h8_downloader.download(r'E:\nc_data', file)

# import ftplib
# import json
# import os
# import time
# import numpy as np
#
# debugLocalDownload = True
# debugDownloadDaily = False
#
# globPersonalTime = [2022, 9, 7]
#
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
#                             np.int16, np.int32, np.int64, np.uint8,
#                             np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, (np.ndarray,)):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
#
# class himawari:
#     ftp = ftplib.FTP()
#
#     def __init__(self):
#         self._url = '/jma/netcdf/'
#         self._save_path = r'E:\nc_data'
#         if debugLocalDownload:
#             self._save_path = r'E:\nc_data'
#         self.ftp.connect('ftp.ptree.jaxa.jp', 21)
#         self.ftp.login('2371016_stu.neu.edu.cn', 'SP+wari8')
#         self._yearNum, self._monNum, self._dayNum = self.dayInit()
#         self._nginx_path = ''
#         print(self.ftp.welcome)  # 显示登录信息
#         self.ftp.voidcmd('TYPE I')
#
#     def run(self):
#         self._nginx_path = ''
#
#         try:
#             if debugDownloadDaily:
#                 self._yearNum, self._monNum, self._dayNum = self.getYesterday(self._yearNum, self._monNum, self._dayNum)
#             else:
#                 self._yearNum = globPersonalTime[0]
#                 self._monNum = globPersonalTime[1]
#                 self._dayNum = globPersonalTime[2]
#             self._yearStr, self._monStr, self._dayStr = self.getDateStr(self._yearNum, self._monNum, self._dayNum)
#             ftp_filePath = self._url + self._yearStr + self._monStr + "/" + self._dayStr + "/"
#             # 从目标路径ftp_filePath将文件下载至本地路径dst_filePath
#             dst_filePath = self._nginx_path + self._save_path + self._yearStr + "/" + self._monStr + "/" + self._dayStr + "/" + "hour" + "/"
#             self.deleteFile(dst_filePath)  # 先删除未下载完成的临时文件
#             print("Local:" + dst_filePath)
#             print("Remote:" + ftp_filePath)
#             self.DownLoadFileTree(dst_filePath, ftp_filePath)
#             if debugDownloadDaily:
#                 self.ftp.quit()
#         except Exception as err:
#             print(err)
#
#     def getYesterday(self, yy, mm, dd):
#         dt = (yy, mm, dd, 9, 0, 0, 0, 0, 0)
#         dt = time.mktime(dt) - 86400
#         yesterdayList = time.strftime("%Y-%m-%d", time.localtime(dt)).split('-')
#         return int(yesterdayList[0]), int(yesterdayList[1]), int(yesterdayList[2])
#
#     def dayInit(self, ):
#         yesterdayList = time.strftime("%Y-%m-%d", time.localtime(time.time())).split('-')
#         return int(yesterdayList[0]), int(yesterdayList[1]), int(yesterdayList[2])
#
#     def getDateStr(self, yy, mm, dd):
#         syy = str(yy)
#         smm = str(mm)
#         sdd = str(dd)
#         if mm < 10:
#             smm = '0' + smm
#         if dd < 10:
#             sdd = '0' + sdd
#         return syy, smm, sdd
#
#     # 删除目录下扩展名为.temp的文件
#     def deleteFile(self, fileDir):
#         if os.path.isdir(fileDir):
#             targetDir = fileDir
#             for file in os.listdir(targetDir):
#                 targetFile = os.path.join(targetDir, file)
#                 if targetFile.endswith('.temp'):
#                     os.remove(targetFile)
#
#     # 下载单个文件，LocalFile表示本地存储路径和文件名，RemoteFile是FTP路径和文件名
#     def DownLoadFile(self, LocalFile, RemoteFile):
#         bufSize = 102400
#         file_handler = open(LocalFile, 'wb')
#         print(file_handler)
#         print('>>>>>>>>>>>>下载文件 %s ... ...' % LocalFile)
#         # 接收服务器上文件并写入本地文件
#         self.ftp.retrbinary('RETR ' + RemoteFile, file_handler.write, bufSize)
#         self.ftp.set_debuglevel(0)
#         file_handler.close()
#         return True
#
#     # 下载整个目录下的文件，LocalDir表示本地存储路径， emoteDir表示FTP路径
#     def DownLoadFileTree(self, LocalDir, RemoteDir):
#         # 如果本地不存在该路径，则创建
#         if not os.path.exists(LocalDir):
#             os.makedirs(LocalDir)
#             # 获取FTP路径下的全部文件名，以列表存储
#         self.ftp.cwd(RemoteDir)
#         RemoteNames = self.ftp.nlst()
#         RemoteNames.reverse()
#         # print("RemoteNames：", RemoteNames)
#         for file in RemoteNames:
#             # 先下载为临时文件Local,下载完成后再改名为nc4格式的文件
#             # 这是为了防止上一次下载中断后，最后一个下载的文件未下载完整，而再开始下载时，程序会识别为已经下载完成
#             Local = os.path.join(LocalDir, file[0:-3] + ".temp")
#             files = file[0:-3] + ".nc"
#             LocalNew = os.path.join(LocalDir, files)
#             '''
#             下载小时文件，只下载UTC时间0时至24时（北京时间0时至24时）的文件
#             下载的文件必须是nc格式
#             若已经存在，则跳过下载
#             '''
#             # 小时数据命名格式示例：H08_20200819_0700_1HARP030_FLDK.02401_02401.nc
#             if (int(file[16:18]) >= 22 or int(file[16:18]) <= 10) and file[21]=='R'and int(file[31])==6:
#                 if not os.path.exists(LocalNew):
#                     print("Downloading the file of %s" % file)
#                     self.DownLoadFile(Local, file)
#                     os.rename(Local, LocalNew)
#                     print("The download of the file of %s has finished\n" % file)
#                     #print("png of the file of %s has finished\n" % png_name)
#                 elif os.path.exists(LocalNew):
#                     print("The file of %s has already existed!\n" % file)
#         self.ftp.cwd("..")
#         return
#
#
# # 主程序
# myftp = himawari()
# if debugDownloadDaily:
#     myftp.run()
# else:
#     yyStart, mmStart, ddStart = input("Start(yy mm dd):").split()
#     yyStart, mmStart, ddStart = int(yyStart), int(mmStart), int(ddStart)
#     yyEnd, mmEnd, ddEnd = input("End(yy mm dd):").split()
#     yyEnd, mmEnd, ddEnd = int(yyEnd), int(mmEnd), int(ddEnd)
#     dtStart = (yyStart, mmStart, ddStart, 9, 0, 0, 0, 0, 0)
#     dtEnd = (yyEnd, mmEnd, ddEnd, 10, 0, 0, 0, 0, 0)
#     timeIndex = time.mktime(dtStart)
#     timeIndexEnd = time.mktime(dtEnd)
#     while timeIndex < timeIndexEnd:
#         indexDayList = time.strftime("%Y-%m-%d", time.localtime(timeIndex)).split('-')
#         globPersonalTime[0] = int(indexDayList[0])
#         globPersonalTime[1] = int(indexDayList[1])
#         globPersonalTime[2] = int(indexDayList[2])
#         print(globPersonalTime)
#         myftp.run()
#         timeIndex = int(timeIndex) + 3600 * 24

