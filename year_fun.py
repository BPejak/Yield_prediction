#®Bane - Biosense Institute
import numpy as np
import pandas as pd
import os
import sys
import shutil
import ogr, gdal, osr
import time
import math
from mpi4py import MPI

#from utils import CreateClippingPolygon, ClipRasters, raster2shape, get_shp_data, get_yield_monitor_data, presek_poligona, border_pixels_drop
from func_pools import SentinelProcess,yieldMonitorData, processGroup, parcelChanging
sys.path.insert(0, "C://Users//Pejak//PycharmProjects//Yield_monitor")
os.environ['GDAL_DATA'] = r'C:/Users/Pejak/anaconda3/envs/Cybele/Library/share/gdal'


#----CREATING DATABASE DONE------------
def CallYear(year,owd):
    comm=MPI.COMM_WORLD ##obtain the communicator
    size = comm.Get_size()
    rank = comm.Get_rank()
   
    lista_slika = os.listdir("D:\\Sentinel-2/Sentinel-2_" + str(year) + "_Austrija/Stack") # list of all images in folder
    lista_parcela = os.listdir("Data\\Shpfiles_Austria_32633_pipeline\\" + str(year))
    if (rank==0 or rank==1):   
        os.makedirs(str(year)+ '_parcels_soybean')
        parcelChanging(lista_parcela,lista_slika,year,owd) ##light (less than 20 sec)
    # #    ########################## BASELINE CALL ##########################################       
        
    #     print('--------PARCEL SHAPE FILE CREATED------------')   

     
   
    num_lista_slika=len(lista_slika)        
    comm.barrier()##wait 
    
   #2 is the number of year, hence processes are grouped into 2               
    SentinelProcess(year,math.floor(rank/2),round(size/2), num_lista_slika) ##for each year, 10 processes     heavy func         
       
    
        
       
    if (rank==0 or rank==1):
        final_data = pd.DataFrame()        
        if os.path.exists(str(year) + '_results'):
            shutil.rmtree(str(year) + '_results')
        os.makedirs(str(year) + '_results')
      
    comm.barrier()##all processes wait here       
    yieldMonitorData (year,math.floor(rank/2),round(size/2)) #2 is the number of year          
    comm.barrier()####all processes wait here   
    if (rank==0 or rank==1):    
        print('--------YIELD PER PIXEL DONE------------')
       
        if os.path.exists(str(year) + '_results_excels'):
            shutil.rmtree(str(year) + '_results_excels')
        os.makedirs(str(year) + '_results_excels')

        list_data = os.listdir(str(year) + "_parcels_soybean") # list of all groups (data groups)    
        processGroup(list_data,year)   
          
            

        final_data = pd.DataFrame()
        list_data = os.listdir(str(year) + "_results_excels")
        for i in range(len(list_data)):
            file_path_sat_data = str(year) + '_results_excels/' + list_data[i]
            xl = pd.ExcelFile(file_path_sat_data)
            satellite_data = xl.parse('Sheet1')
            satellite_data = satellite_data.drop(['ID.1', 'ID.2', 'Latitude.1', 'Longitude.1', 'ID.3', 'Latitude.2', 'Longitude.2'], axis=1)
            satellite_data['Parcel_ID'] = str(list_data[i])[:-5]
            satellite_data['Year'] = str(year)

            final_data = pd.concat([final_data, satellite_data], axis=0) # stack all data in one file###shared data, needs a lock

        final_data.to_excel(str(year) + '_final_dataset_V_MPI.xlsx', index=False)
        print (str(year)+'_final_dataset_V_MPI.xlsx', 'file created!!!')#test
