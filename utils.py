# ®Bane - Biosense Institute
#!/bin/env python
# -*- coding: utf-8 -*-

import shutil
#from  osgeo import ogr###Zhou
#import gdal, osr, os ##Zhou
import matplotlib.pyplot as plt
import ogr, gdal, osr, os, gdalnumeric
import numpy as np
import itertools
import pandas as pd
import os
import pickle
import sys
import csv
import geopandas as gp
import glob
import subprocess
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.ensemble import RandomForestRegressor
#from math import sqrt
#from scipy.spatial import distance
#import scipy.spatial.distance
#import scipy
import cv2
import math
import copy
import joblib
#sys.path.insert(0, "/shared_folder/pilot1/")
sys.path.insert(0, "C://Users//Pejak//PycharmProjects//Yield_monitor")
os.environ['GDAL_DATA'] = r'C:/Users/Pejak/anaconda3/envs/Cybele/Library/share/gdal'

#sys.path.insert(0, "/shared_folder/pilot1/")
#sys.path.insert(0, "/home/ZNWL/code/pilot1/new_data")
#os.environ['GDAL_DATA'] = r'/usr/local/conda3/share/gdal'
#os.environ['PROJ_LIB'] = r'/usr/local/conda3/share/proj'
#os.environ['GDAL_DATA'] = r'/usr/local/conda3/envs/pilotoneEnv/share/gdal' ##specify the path for gdal and proj
#os.environ['PROJ_LIB'] = r'/usr/local/conda3/envs/pilotoneEnv/share/proj'
#### RASTER CLIP / PARCEL CLIP #######

########## V5 !!!!!!!!!!!
def ClipRasterWithPolygon(rasterPath, polyPath, outPath):
   #os.system("ogr2ogr -t_srs EPSG:32633 %s %s" % (polyPath,outPath))
   os.system("gdalwarp -dstnodata -9999 -q -cutline " + polyPath + " -crop_to_cutline " + " -of GTiff " + rasterPath + " " + outPath) #-9999 tamo gde mi slika ne treba!

########## V6 !!!!!!!!!!!
# def ClipRasterWithPolygon(inraster, inshape, outraster):
#     # open raster and get its georeferencing information
#     dsr = gdal.Open(inraster, gdal.GA_ReadOnly)
#     gt = dsr.GetGeoTransform()
#     srr = osr.SpatialReference()
#     srr.ImportFromWkt(dsr.GetProjection())
#
#     # open vector data and get its spatial ref
#     dsv = ogr.Open(inshape)
#     lyr = dsv.GetLayer(0)
#     srv = lyr.GetSpatialRef()
#
#     # make object that can transorm coordinates
#     ctrans = osr.CoordinateTransformation(srv, srr)
#
#     lyr.ResetReading()
#     ft = lyr.GetNextFeature()
#     while ft:
#         # read the geometry and transform it into the raster's SRS
#         geom = ft.GetGeometryRef()
#         #geom.Transform(ctrans) ###########################################
#         # get bounding box for the transformed feature
#         minx, maxx, miny, maxy = geom.GetEnvelope()
#
#         # compute the pixel-aligned bounding box (larger than the feature's bbox)
#         left = minx - (minx - gt[0]) % gt[1]
#         right = maxx + (gt[1] - ((maxx - gt[0]) % gt[1]))
#         bottom = miny + (gt[5] - ((miny - gt[3]) % gt[5]))
#         top = maxy - (maxy - gt[3]) % gt[5]
#         subprocess.call(['gdalwarp','-q', inraster, outraster,'-cutline', inshape,'-tr', str(abs(gt[1])), str(abs(gt[5])),
#                          '-te', str(left), str(bottom), str(right), str(top)])
#         ft = lyr.GetNextFeature()


########## V7 !!!!!!!!!!!
# def refine_pixel_geo_coordinate(point,pixelWidth,pixelHeight,flag, crop_offset):
#
#     point = list(point)
#     if flag == 0:
#          point[0] = np.floor((point[0]/ pixelWidth)) * pixelWidth
#          point[1] = np.floor((point[1] / np.abs(pixelHeight))) * np.abs(pixelHeight)
#          if crop_offset != 0:
#              point[0] -= (crop_offset * pixelWidth)
#              point[1] -= (crop_offset * np.abs(pixelHeight))
#     else:
#         point[0] = np.ceil((point[0] / pixelWidth)) * pixelWidth
#         point[1] = np.ceil((point[1] / np.abs(pixelHeight))) * np.abs(pixelHeight)
#         if crop_offset != 0:
#             point[0] += (crop_offset * pixelWidth)
#             point[1] += (crop_offset * np.abs(pixelHeight))
#
#     return tuple(point)
#
#
# def ClipRasterWithPolygon(in_raster, shapefile, out_raster): #create_mask_from_shp(in_raster,in_reliable_mask, shapefile, crop_offset):
#
#     #open raster file and reproject it if needed
#     raster = gdal.Open(in_raster)# read the raster file from which we use projection and create a grid on which we gonna interpolate
#     #reliable_reflectance_mask = gdalnumeric.LoadFile(in_reliable_mask)
#     image =  gdalnumeric.LoadFile(in_raster)
#     crop_offset = 0
#     #read transform from raster
#     geo_transform = raster.GetGeoTransform()
#     xOrigin = geo_transform[0]
#     yOrigin = geo_transform[3]
#     pixelWidth = geo_transform[1]
#     pixelHeight = geo_transform[5]
#
#     #read vector and reproject to raster projection
#     border_shape = ogr.Open(shapefile)
#     lyr = border_shape.GetLayer()
#     feat = lyr.GetFeature(0)
#     geom = feat.GetGeometryRef()
#     #geom = #ogr.CreateGeometryFromWkt(wkt)
#
#     minX, maxX, minY, maxY = geom.GetEnvelope()
#     box_start = (minX,minY)
#     box_end = (maxX,maxY)
#     box_start = refine_pixel_geo_coordinate(box_start, pixelWidth, pixelHeight, 0, crop_offset)
#     box_end = refine_pixel_geo_coordinate(box_end, pixelWidth, pixelHeight, 1, crop_offset)
#
#     pts_geom = geom.GetGeometryRef(0)
#     pts = np.zeros(shape=(pts_geom.GetPointCount(), 2), dtype=np.int32)
#
#     for i in range(0,pts_geom.GetPointCount()):
#         pt = pts_geom.GetPoint(i)
#         pts[i, 0] = np.int32((pt[0] - xOrigin) / pixelWidth)
#         pts[i, 1] = np.int32((pt[1] - yOrigin) / pixelHeight)
#
#     x_min = np.int32((box_start[0] - xOrigin) / pixelWidth)
#     y_min = np.int32((box_end[1] - yOrigin) / pixelHeight)
#     x_max = np.int32((box_end[0] - xOrigin) / pixelWidth)
#     y_max = np.int32((box_start[1] - yOrigin) / pixelHeight)
#
#     cols = raster.RasterXSize
#     rows = raster.RasterYSize
#     mask = np.zeros(shape=(rows,cols), dtype='uint8')
#     cv2.fillPoly(mask, [pts], 1)
#
#     #ROI = tuple([y_min,y_max,x_min,x_max])
#     croped_mask = copy.deepcopy(mask[y_min:y_max,x_min:x_max])
#     croped_image = copy.deepcopy(image[:,y_min:y_max,x_min:x_max])
#     #croped_reliable_reflectance_mask = copy.deepcopy(reliable_reflectance_mask[y_min:y_max,x_min:x_max])
#
#     idx = (croped_mask == 0)
#     #croped_image[idx] = chex[idx]
#     image_final = croped_image * croped_mask
#     croped_geo_transform = copy.deepcopy(geo_transform)
#     tmp_list = list(croped_geo_transform)
#
#     #shift Origin
#     tmp_list[0] = box_start[0]
#     tmp_list[3] = box_end[1]
#     croped_geo_transform = tuple(tmp_list)
#     num_bands = 12
#     save_image_as_geotiff(image_final, num_bands, in_raster, out_raster,croped_geo_transform)
#     a=4


def save_image_as_geotiff(image, num_bands, in_raster, out_raster,croped_geo_transform):

    raster = gdal.Open(in_raster)
    driver = gdal.GetDriverByName("GTiff")
    cols = np.shape(image)[2]#cols = raster.RasterXSize
    rows = np.shape(image)[1]#rows = raster.RasterYSize
    # print driver
    if image.dtype =='uint16':
        data_type = gdal.GDT_UInt16
    elif image.dtype =='uint8':
        data_type = gdal.GDT_Byte

    out_raster = driver.Create(out_raster, cols, rows, num_bands, data_type, ['COMPRESS=NONE', "INTERLEAVE=BAND", "TILED=YES", 'NUM_THREADS=ALL_CPUS'])
    if out_raster is None:
        print('Not defined path for saving stacked image !!!')
        sys.exit(1)

    # georeference the image and set the projection
    #out_raster.SetGeoTransform(raster.GetGeoTransform())
    out_raster.SetGeoTransform(croped_geo_transform)
    out_raster.SetProjection(raster.GetProjection())

    # # write the data
    with joblib.parallel_backend("threading"):
        if num_bands > 1:
            for i in range(num_bands):
                out_raster.GetRasterBand(i+1).WriteArray(image[i,:,:]) #image[:,:,i]
        else:
            out_raster.GetRasterBand(1).WriteArray(image)

        out_raster.FlushCache()
        out_raster = None
        A=4


#def CreateClippingPolygon(inPath, field, i, owd, link_path):
def CreateClippingPolygon(inPath, field, i, link_path):    
   # os.chdir(owd)
    owd=os.getcwd()#Zhou
    driverSMP = ogr.GetDriverByName("ESRI Shapefile")
    print('in path--' ,inPath)
    print ('current directory in CCP',owd)
    ds = driverSMP.Open(inPath, 0)

    if ds is None:
        print('layer not open')
    lyr = ds.GetLayer()
    srs = ogr.osr.SpatialReference()

    #srs.ImportFromEPSG(32633)

    for feature in lyr:
        fieldVal = feature.GetField(field)
       # os.chdir(owd + link_path + "/")
        os.mkdir(owd + link_path + "/"+"ClippingFeatures" + str(i) + "/" + str(fieldVal))
        outds = driverSMP.CreateDataSource(owd + link_path + "/"+"ClippingFeatures" + str(i) + "/" + str(fieldVal) + "/clip.shp")
        outlyr = outds.CreateLayer(owd + link_path + "/"+str(fieldVal) + "/clip.shp", srs, geom_type=ogr.wkbPolygon )

        outDfn = outlyr.GetLayerDefn()
        ingeom = feature.GetGeometryRef()
        outFeat = ogr.Feature(outDfn)
        outFeat.SetGeometry(ingeom)
        outlyr.CreateFeature(outFeat)

#def ClipRasters(inPath, field, i, owd, link_path, year):
def ClipRasters(inPath, field, i, link_path, year):
    #os.chdir(owd)
    owd=os.getcwd()#Zhou
    driverSMP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSMP.Open(inPath)
    if ds is None:
        print('layer not open')
    lyr = ds.GetLayer()

    lista_slika = os.listdir("D:\\Sentinel-2/Sentinel-2_"+ str(year) +"_Austrija/Stack")
    slike_u_folderu = os.listdir('D:\\Sentinel-2/Sentinel-2_'+ str(year) +'_Austrija/Stack/' + str(lista_slika[i]))

    stakovana_slika = str(slike_u_folderu[0]) #python 9 stacked image in folder (10) !!!!!!!!!
    for feature in lyr:
        fieldVal = feature.GetField(field)

        inputProductPath = "D:\\Sentinel-2/Sentinel-2_"+ str(year) +"_Austrija/Stack/" + str(lista_slika[i]) + "/" + str(stakovana_slika) ######### change image
        ClipRasterWithPolygon(inputProductPath, owd + link_path + "/ClippingFeatures" + str(i) + "/" + str(fieldVal), owd + link_path + "/ClippingFeatures"  + str(i) + "/" + str(fieldVal)+"/dem.tif")


######### CREATE IMAGE POLYGONS #######################


def pixelOffset2coord(raster, xOffset,yOffset):
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset + pixelWidth*0.5
    coordY = originY+pixelHeight*yOffset + pixelHeight*0.5
    return coordX, coordY

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    return array

def get_pixel_value(raster,Xcoord,Ycoord):
    channel_all = pd.DataFrame()
    for i in range(1,13):        # take values for all 12 channels, Number of channels
        band = raster.GetRasterBand(i)  # 1-based index
        data = band.ReadAsArray()

        transform = raster.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        # get individual pixel values
        xOffset = int((Xcoord - xOrigin) / pixelWidth)
        yOffset = int((Ycoord - yOrigin) / pixelHeight)
        value = pd.Series(data[yOffset][xOffset])
        channel_all = pd.concat([channel_all, value], axis = 0) # spajam da bih dobio svih 13 kanala
    return(channel_all)


def array2shp(array,outSHPfn,rasterfn):

    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    pixelWidth = geotransform[1]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)

    # wkbPoint
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, srs, geom_type=ogr.wkbMultiPolygon) #wkbPoint
    featureDefn = outLayer.GetLayerDefn()
    outLayer.CreateField(ogr.FieldDefn("Latitude")) #, ogr.OFTInteger))
    outLayer.CreateField(ogr.FieldDefn("Longitude"))#, ogr.OFTInteger))
    outLayer.CreateField(ogr.FieldDefn("ID"))
    outLayer.CreateField(ogr.FieldDefn("Ch1"))
    outLayer.CreateField(ogr.FieldDefn("Ch2"))
    outLayer.CreateField(ogr.FieldDefn("Ch3"))
    outLayer.CreateField(ogr.FieldDefn("Ch4"))
    outLayer.CreateField(ogr.FieldDefn("Ch5"))
    outLayer.CreateField(ogr.FieldDefn("Ch6"))
    outLayer.CreateField(ogr.FieldDefn("Ch7"))
    outLayer.CreateField(ogr.FieldDefn("Ch8"))
    outLayer.CreateField(ogr.FieldDefn("Ch9"))
    outLayer.CreateField(ogr.FieldDefn("Ch10"))
    outLayer.CreateField(ogr.FieldDefn("Ch11"))
    outLayer.CreateField(ogr.FieldDefn("Ch12"))


###############################################################################
    # Dodato za lat lon
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(raster.GetProjectionRef())

    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""


    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)


    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)

#############################################################################

    # array2dict
    row_count = array.shape[0]
    id_num = 1

    for ridx, row in enumerate(array):
        if ridx % 100 == 0:
            print("{0} of {1} rows processed".format(ridx, row_count))

        for cidx, value in enumerate(row):
            if value == 0: #-9999:
                pass
            else:

                Xcoord, Ycoord = pixelOffset2coord(raster,cidx,ridx)

                originX = geotransform[0]
                originY = geotransform[3]
                pixelWidth = geotransform[1]
                pixelHeight = geotransform[5]

                # get individual pixel values
                xOffset = int((Xcoord - originX) / pixelWidth)
                yOffset = int((Ycoord - originY) / pixelHeight)
                coord1 = (originX + pixelWidth * xOffset, originY + pixelHeight * yOffset)
                coord2 = (originX + pixelWidth * xOffset + pixelWidth, originY + pixelHeight * yOffset + pixelHeight)
                coord3 = (originX + pixelWidth * xOffset, originY + pixelHeight * yOffset + pixelHeight)
                coord4 = (originX + pixelWidth * xOffset + pixelWidth, originY + pixelHeight * yOffset)
                coord5 = (originX + pixelWidth * xOffset, originY + pixelHeight * yOffset)

                wkt1 = 'POLYGON ((' + str(coord1[0]) + ' ' + str(coord1[1]) + ', ' + str(
                    coord4[0]) + ' ' + str(coord4[1]) + ', ' + str(coord2[0]) + ' ' + str(
                    coord2[1]) + ', ' + str(coord3[0]) + ' ' + str(coord3[1])+ ', ' + str(coord5[0])+ ' ' + str(coord5[1]) + '))'
                poly1 = ogr.CreateGeometryFromWkt(wkt1)
                #poly1.Transform(transform)

                latlong = transform.TransformPoint(Xcoord, Ycoord)

                #######################
                channel_all = get_pixel_value(raster,Xcoord,Ycoord)
                channel_all = channel_all.reset_index(drop=True)

                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(poly1)
                outFeature.SetField("Latitude", float(latlong[1]))
                outFeature.SetField("Longitude", float(latlong[0]))
                outFeature.SetField("ID", id_num)
                outFeature.SetField("Ch1", float(channel_all.loc[0]))
                outFeature.SetField("Ch2", float(channel_all.loc[1]))
                outFeature.SetField("Ch3", float(channel_all.loc[2]))
                outFeature.SetField("Ch4", float(channel_all.loc[3]))
                outFeature.SetField("Ch5", float(channel_all.loc[4]))
                outFeature.SetField("Ch6", float(channel_all.loc[5]))
                outFeature.SetField("Ch7", float(channel_all.loc[6]))
                outFeature.SetField("Ch8", float(channel_all.loc[7]))
                outFeature.SetField("Ch9", float(channel_all.loc[8]))
                outFeature.SetField("Ch10", float(channel_all.loc[9]))
                outFeature.SetField("Ch11", float(channel_all.loc[10]))
                outFeature.SetField("Ch12", float(channel_all.loc[11]))
                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
                id_num = id_num + 1


def raster2shape(rasterfn,outSHPfn):
    array = raster2array(rasterfn)
    array2shp(array,outSHPfn,rasterfn)

def get_shp_data(file_path_shp):

    myList = []

    driverSMP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSMP.Open(file_path_shp)
    if ds is None:
        print('layer not open')
    lyr = ds.GetLayer()

    for feature in lyr:
        Latitude = feature.GetField('Latitude')       
        Longitude = feature.GetField('Longitude')       
        parcel_ID = feature.GetField('ID')
        Ch1 = feature.GetField('Ch1')
        Ch2 = feature.GetField('Ch2')
        Ch3 = feature.GetField('Ch3')
        Ch4 = feature.GetField('Ch4')
        Ch5 = feature.GetField('Ch5')
        Ch6 = feature.GetField('Ch6')
        Ch7 = feature.GetField('Ch7')
        Ch8 = feature.GetField('Ch8')
        Ch9 = feature.GetField('Ch9')
        Ch10 = feature.GetField('Ch10')
        Ch11 = feature.GetField('Ch11')
        Ch12 = feature.GetField('Ch12')

        myList.append([Latitude, Longitude, parcel_ID,Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8,Ch9,Ch10,Ch11,Ch12])
   
    data = pd.DataFrame(myList, columns=['Latitude', 'Longitude', 'ID','Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8','Ch9','Ch10','Ch11','Ch12'])

    return(data)


############### YIELD PER PIXEL - INTERSECTION POLYGONS/PIXELS ##################

def jednaShp(file_path_shp_yield, data_all, list_data, m, year):

    infile = ogr.Open(file_path_shp_yield,1)
    dataLayer = infile.GetLayerByIndex(0)

    path = str(year)+"_parcels_soybean/" + str(list_data) + "/ClippingFeatures0/" + str(m) +"/clip.shp",
    print(path)
    shpDriver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource_ = shpDriver.Open(path[0])
    lyr_parcel = dataSource_.GetLayer()
    polyyy = lyr_parcel.GetNextFeature()


    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    target = osr.SpatialReference()
    target.ImportFromEPSG(32633) #32633

    transform = osr.CoordinateTransformation(source, target)


    listaPoly = []

    outLayer = None
    outDataSource = None
    outShapefile = str(year)+'_results/YM_polygons_' + str(list_data) + '_ID' + str(m)+ '.shp'
    outDataSource = shpDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer(outShapefile, target, geom_type=ogr.wkbPolygon)

    geom = polyyy.GetGeometryRef()
    geom.Transform(transform)

    idField = ogr.FieldDefn('DryYield', ogr.OFTReal)
    outLayer.CreateField(idField)
    idField = ogr.FieldDefn('OldArea', ogr.OFTReal)
    outLayer.CreateField(idField)
    idField = ogr.FieldDefn('NewArea', ogr.OFTReal)
    outLayer.CreateField(idField)
    idField = ogr.FieldDefn('Yield_kg', ogr.OFTReal)
    outLayer.CreateField(idField)

    print(outShapefile)
   # dataLayer.ResetReading()##zhou
    for feature in dataLayer:        
        Latitude = feature.GetField('Latitude')
        Longitude = feature.GetField('Longitude')        
        ugao = feature.GetField('Heading')
        pomerja = feature.GetField('Distance')
        sirina = feature.GetField('Width')
        dryYield = feature.GetField("Dry Yield")
        flow = feature.GetField("Flow Soybe")
        speed = feature.GetField("Speed")

        #
        try:
            point = ogr.Geometry(ogr.wkbPoint)
            cord_transf = transform.TransformPoint(Longitude,Latitude)
            point.AddPoint(float(cord_transf[0]), float(cord_transf[1]), float(cord_transf[2]))
            if geom.Intersect(point):
                if sirina > 0 and pomerja > 0:


                    poly = nekoCudo(point, sirina, pomerja, ugao/360*2*math.pi)
                    areaOld = poly.Area()
                    if geom.Intersect(poly):
                        if poly is not None:
                            poly = geom.Intersection(poly)
                            for oldPoly in listaPoly:
                                if poly.Intersect(oldPoly):
                                    tempPoly = geom.Intersection(poly)
                                    if tempPoly is not None:
                                        poly = tempPoly
                    listaPoly.append(poly.Clone())
                    a = 5
                    areaNew = poly.Area()
                    kkk = 88
                    outLayerDefn = outLayer.GetLayerDefn()
                    outFeature = ogr.Feature(outLayerDefn)
                    outFeature.SetField('DryYield', dryYield)
                    outFeature.SetField('OldArea', areaOld)
                    outFeature.SetField('NewArea', areaNew)
                    outFeature.SetField('Yield_kg', dryYield*(areaNew/10000))
                    outFeature.SetGeometry(poly.Clone())
                    outLayer.CreateFeature(outFeature)
                    outFeature.Destroy()

        except:
            print('error')
            #print('error')
        #a=2

############### DRAW YIELD POLYGONS ##################

def nekoCudo(point, sirina, pomerja, ugao):

    x = point.GetX()
    y = point.GetY()

    point1 = ogr.Geometry(ogr.wkbPoint)
    point1.AddPoint(x - sirina / 2 * math.cos(ugao), y + sirina / 2 * math.sin(ugao),0)

    point2 = ogr.Geometry(ogr.wkbPoint)
    point2.AddPoint(x + sirina / 2 * math.cos(ugao), y - sirina / 2 * math.sin(ugao), 0)

    pomerajX = pomerja * math.sin(math.pi - ugao)
    pomerajY = pomerja * math.cos(math.pi - ugao)

    point3 = ogr.Geometry(ogr.wkbPoint)
    point3.AddPoint(point2.GetX() - pomerajX , point2.GetY() + pomerajY, 0)

    point4 = ogr.Geometry(ogr.wkbPoint)
    point4.AddPoint(point1.GetX() - pomerajX , point1.GetY() + pomerajY, 0)


    outRing = ogr.Geometry(ogr.wkbLinearRing)
    outRing.AddPoint(point1.GetX(), point1.GetY())
    outRing.AddPoint(point2.GetX(), point2.GetY())
    outRing.AddPoint(point3.GetX(), point3.GetY())
    outRing.AddPoint(point4.GetX(), point4.GetY())
    outRing.AddPoint(point1.GetX(), point1.GetY())

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(outRing)

    return poly


def get_yield_monitor_data(file_path_shp_yield, data_all,  list_data, m, year):
    myList = []

    driverSMP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSMP.Open(file_path_shp_yield)
    if ds is None:
        print('layer not open')
    lyr = ds.GetLayer()

    Hub_Latitudes = data_all['Latitude'].T.iloc[0]
    Hub_Longitudes = data_all['Longitude'].T.iloc[0]
    Hub_ID = data_all['ID'].T.iloc[0]

    data_hubs_ = pd.concat([Hub_Latitudes,Hub_Longitudes], axis=1)
    data_hubs_ = [tuple(x) for x in data_hubs_.values]
    data_hubs = pd.concat([Hub_ID, Hub_Latitudes, Hub_Longitudes], axis=1)

    jednaShp(file_path_shp_yield, data_all, list_data, m, year) #######################
    a=3



############### CREATING DATABASE ##################


def border_pixels_drop(file_path_shp):
    df = gp.read_file(file_path_shp)
    if np.size(df.Ch1) > 57: ########################## BILO 30
        df_pomocni = df
        for index, country in df.iterrows():
            neighbors = df[~df.geometry.disjoint(country.geometry.buffer(1))].ID.tolist()
            neighbors = [name for name in neighbors if country.ID != name]
            if len(neighbors) != 8:
                df_pomocni = df_pomocni.drop(index)

        # Izlazni SHP file sa pixel-poligonima i uklonjenim ivicnim pikselima. Treba podesiti odgovarajucu putanju.
        df_pomocni.to_file(file_path_shp)
    else:
        pass

def presek_poligona(lyr_pixel,lyr_polygon):
    myList = []
    for featrure_pixels in lyr_pixel:
        ID = featrure_pixels.GetField("ID")
        lyr_polygon.ResetReading()
        geomPixel = featrure_pixels.GetGeometryRef().Clone()
        lyr_polygon.SetSpatialFilter(geomPixel)
        yieldCum = 0
        areaCum = 0
        for featrure_polygons in lyr_polygon:
            geomPoligon = featrure_polygons.GetGeometryRef().Clone()
            if geomPoligon.Intersect(geomPixel):
                Yield_kg = featrure_polygons.GetField("Yield_kg")
                area_polygons = featrure_polygons.GetField("NewArea")
                geomInt = geomPoligon.Intersection(geomPixel)

                if geomInt is not None:
                    area_intersec = geomInt.Area()
                    procent = area_intersec / area_polygons
                    yieldCum += Yield_kg*procent
                    areaCum += area_intersec

        if(areaCum > 0):

            myList.append([ID, yieldCum, areaCum])
        else:
            myList.append([ID, str(None), str(None)])
    data = pd.DataFrame(myList, columns=['ID', 'Yield', 'Area'])
    return(data)


