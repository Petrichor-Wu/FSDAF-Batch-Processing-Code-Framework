import numpy as np
from osgeo import gdal
import os


def read_raster(infile):
    """
    读取栅格文件并返回 NumPy 数组 (nb, rows, cols)。
    """
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    fp = gdal.Open(infile)
    cols = fp.RasterXSize
    rows = fp.RasterYSize
    nb = fp.RasterCount
    if nb == 1:
        band = fp.GetRasterBand(1)
        data = band.ReadAsArray()
        data = data.reshape(1, rows, cols)
        band.GetScale()
        band.GetOffset()
        band.GetNoDataValue()
    else:
        data = np.zeros([nb, rows, cols])
        for i in range(0, nb):
            band = fp.GetRasterBand(i + 1)
            data[i, :, :] = band.ReadAsArray()
            band.GetScale()
            band.GetOffset()
            band.GetNoDataValue()

    # 【修复】显式关闭文件句柄，释放文件锁
    fp = None

    return rows, cols, data


def read_raster_new(infile):
    """
    读取栅格文件并返回 NumPy 数组 (rows, cols, nb)。
    """
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    fp = gdal.Open(infile)
    cols = fp.RasterXSize
    rows = fp.RasterYSize
    nb = fp.RasterCount
    data = np.zeros([rows, cols, nb])
    for i in range(0, nb):
        band = fp.GetRasterBand(i + 1)
        data[:, :, i] = band.ReadAsArray()
        band.GetScale()
        band.GetOffset()
        band.GetNoDataValue()

    # 【修复】显式关闭文件句柄，释放文件锁
    fp = None

    return rows, cols, data


def writeimage(bands, path, in_ds_path):
    """
    将 NumPy 数组写入到新的栅格文件，并复制源文件的地理参考信息。
    """
    suffix = os.path.splitext(in_ds_path)[-1]
    in_ds = gdal.Open(in_ds_path)

    if bands is None or len(bands) == 0:
        in_ds = None
        return

    band1 = bands[0]
    img_width = band1.shape[1]
    img_height = band1.shape[0]
    num_bands = len(bands)

    datatype = gdal.GDT_Float32  # 统一用 Float32，避免类型冲突

    if suffix.lower() == '.tif':
        driver = gdal.GetDriverByName("GTiff")
    else:
        driver = gdal.GetDriverByName("ENVI")

    # 使用 MEM 驱动先写入内存，避免文件锁
    mem_ds = gdal.GetDriverByName("MEM").Create('', img_width, img_height, num_bands, datatype)
    for i in range(num_bands):
        mem_ds.GetRasterBand(i + 1).WriteArray(bands[i])

    # 再复制到磁盘
    driver.CreateCopy(path, mem_ds, strict=0)

    # 设置地理信息
    geoTransform = in_ds.GetGeoTransform()
    proj = in_ds.GetProjection()
    out_ds = gdal.Open(path, gdal.GA_Update)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(proj)

    # 强制关闭所有句柄
    mem_ds = None
    out_ds = None
    in_ds = None