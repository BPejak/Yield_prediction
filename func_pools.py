#testing code 
import numpy as np
import pandas as pd
import os
import shutil
import ogr, gdal, osr
import time
import sys

from utils import CreateClippingPolygon, ClipRasters, raster2shape, get_shp_data, get_yield_monitor_data, presek_poligona, border_pixels_drop
sys.path.insert(0, "C://Users//Pejak//PycharmProjects//Yield_monitor")
os.environ['GDAL_DATA'] = r'C:/Users/Pejak/anaconda3/envs/Cybele/Library/share/gdal'

def parcelChanging(lista_parcela,lista_slika,year,owd):
    for j in lista_parcela: ## j parcel changing 
        if j.__contains__('.shp'):       
            dir_file = str(j[:-14])
            if os.path.exists(str(year)+ '_parcels_soybean/'+dir_file):
                shutil.rmtree(str(year)+ '_parcels_soybean/'+dir_file)
            os.makedirs(str(year)+ '_parcels_soybean/'+dir_file)
           

            for i in range(0, len(lista_slika)): ## i image changing 
                link_path = '/' + str(year) + '_parcels_soybean/' + dir_file      
                dir_img = "ClippingFeatures" + str(i)
                if os.path.exists(owd + link_path + "/"+dir_img):
                    shutil.rmtree(owd + link_path + "/"+dir_img)
                os.makedirs(owd + link_path + "/"+dir_img)

                CreateClippingPolygon("Data/Shpfiles_Austria_32633_pipeline/" +  str(year) + "/" + str(j), "id", i, link_path)# change iteratively: year - parcel - image. Define the polygon which wiil be used for cropping images
                ClipRasters("Data/Shpfiles_Austria_32633_pipeline/" +  str(year) + "/" + str(j), "id", i, link_path, year) # cropping images
        else:
            pass

def SentinelProcess(year,k,size,num_array):    
    #for k in range(0, len(lista_slika)): ## image changing, len() - number of images
    my_pos=k   
    list_data = os.listdir(str(year)+ '_parcels_soybean/') ## list of gruoup data  
    while my_pos<num_array:      
        for i in range(len(list_data)): ## i group of parcel changing
            parcel_number = os.listdir(str(year)+ '_parcels_soybean/' + list_data[i] + '/ClippingFeatures'+ str(my_pos)) # list of all parcels in particular folder , 2 is year 2018 and 2019
            for j in range(len(parcel_number)): # j parcel number (parcel_ID)
                rasterfn = str(year)+ '_parcels_soybean/' + list_data[i] + '/ClippingFeatures' + str(my_pos) + '/' + parcel_number[j] +'/dem.tif'   # ClippingFeatures0 1 2
                outSHPfn = str(year)+ '_parcels_soybean/' + list_data[i] + '/parcel_id_'+ parcel_number[j] +'-Pixel_centroids-' + str(my_pos) + '.shp' #### output is a shp file
                raster2shape(rasterfn,outSHPfn) # raster cropped images to shp file (pixel_polygons)
        my_pos=my_pos+size
       
                                  



#heavy func. size: size of the processes in this group
def yieldMonitorData (year,k,size):
    list_data = os.listdir(str(year) + "_parcels_soybean") # list of all groups data
    #for n in range(len(list_data)):    
    my_pos=k
    while my_pos<len(list_data):
        link = os.listdir(str(year) +'_parcels_soybean/' + list_data[my_pos] + '/ClippingFeatures0')    
        for m in range(1, len(link) + 1):
            # for m in link:
            file_path_shp_0 = str(year) + '_parcels_soybean/' + list_data[my_pos] + '/parcel_id_' + str(m) + '-Pixel_centroids-0.shp'
            data_0 = get_shp_data(file_path_shp_0)
            file_path_shp_1 = str(year) + '_parcels_soybean/' + list_data[my_pos] + '/parcel_id_' + str(m) + '-Pixel_centroids-1.shp'
            data_1 = get_shp_data(file_path_shp_1)
            file_path_shp_2 = str(year) + '_parcels_soybean/' + list_data[my_pos] + '/parcel_id_' + str(m) + '-Pixel_centroids-2.shp'
            data_2 = get_shp_data(file_path_shp_2)
            # lepim podatke za obe slike u jedan dataframe
            data_all = pd.concat([data_0, data_1, data_2], axis=1)
            
            file_path_shp_yield = 'Austria/' + list_data[my_pos] + '/TackeSHP/' + list_data[my_pos] + '_ID' + str(m) + '.shp'  # ovde menjam ID parcele za yield monitor
            print(file_path_shp_yield)

            yield_monitor_data = get_yield_monitor_data(file_path_shp_yield, data_all, list_data[my_pos], m, year)
        my_pos=my_pos+size
        



def processGroup(list_data,year):
    for n in range(len(list_data)): # take every group iteratively
        link = os.listdir(str(year) + '_parcels_soybean/' + list_data[n] + '/ClippingFeatures0')
        for m in range(1, len(link) + 1):
            final_data = pd.DataFrame()
            file_path_shp_0 = str(year) + '_parcels_soybean/' + list_data[n] + '/parcel_id_' + str(m) + '-Pixel_centroids-0.shp'
            print(file_path_shp_0)
            border_pixels_drop(file_path_shp_0)
            data_0 = get_shp_data(file_path_shp_0)
            file_path_shp_1 = str(year) + '_parcels_soybean/' + list_data[n] + '/parcel_id_' + str(m) + '-Pixel_centroids-1.shp'
            border_pixels_drop(file_path_shp_1)
            data_1 = get_shp_data(file_path_shp_1)
            file_path_shp_2 = str(year) + '_parcels_soybean/' + list_data[n] + '/parcel_id_' + str(m) + '-Pixel_centroids-2.shp'
            border_pixels_drop(file_path_shp_2)
            data_2 = get_shp_data(file_path_shp_2)

            # lepim podatke za obe slike u jedan dataframe
            data_all = pd.concat([data_0, data_1, data_2], axis=1)

            path_to_shp_pixel = str(year) + '_parcels_soybean/' + list_data[n] + '/parcel_id_' + str(m) + '-Pixel_centroids-0.shp'
            path_to_shp_polygon = str(year) + "_results/YM_polygons_" + list_data[n] + "_ID" + str(m) + ".shp"

            pixel_shape = ogr.Open(path_to_shp_pixel)
            polygon_shape = ogr.Open(path_to_shp_polygon)
            lyr_pixel = pixel_shape.GetLayer()
            lyr_polygon = polygon_shape.GetLayer()

            yields_df = presek_poligona(lyr_pixel, lyr_polygon)

            final_data = pd.concat([yields_df, data_all], axis=1)

            final_data.to_excel(str(year) + '_results_excels/' + list_data[n] + '_ID' + str(m) + '.xlsx', index=False)
            data_all = pd.DataFrame()
            final_data = pd.DataFrame()
            yields_df = pd.DataFrame()
    
