#®Bane - Biosense Institute
import numpy as np
import pandas as pd
import os
import shutil
import ogr, gdal, osr
import time
from mpi4py import MPI
from year_fun import CallYear
os.environ['OPENMPI_HOME'] = "C:\\Program Files\\Microsoft MPI\\Bin"

# def _init_openmpi():
#     """Pre-load libmpi.dll and register OpenMPI distribution."""
#     import os
#     import ctypes
#     if os.name != 'nt' or 'OPENMPI_HOME' in os.environ:
#         return
#     try: 0,

#         openmpi_home = os.path.abspath(os.path.dirname(__file__))
#         openmpi_bin = os.path.join(openmpi_home, 'bin')
#         os.environ['OPENMPI_HOME'] = openmpi_home
#         os.environ['PATH'] = ';'.join((openmpi_bin, os.environ['PATH']))
#         ctypes.cdll.LoadLibrary(os.path.join(openmpi_bin, 'libmpi.dll'))
#     except Exception:
#         pass
#
# _init_openmpi()

if __name__ == '__main__':
    start_time = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
   
   
    years = [2018, 2019, 2020]    ##processes are grouped into the number of the year (20 max)
    owd = os.getcwd()  
    # for i in range (len(years)): ## year changing        
    #     if i==rank%len(years):            
    #         CallYear(years[i],owd) 
            
    if rank % len(years) == 0:
        CallYear(years[0],owd)
    if rank % len(years) == 1:
        CallYear(years[1],owd)
    if rank % len(years) == 2:
        CallYear(years[2], owd)
            ########################## BASELINE CALL ##########################################
     
    print('--------CREATING FINAL DATABASE DONE------------')
    comm.barrier() ##all processes wait the rest to finish
    if rank ==0: ##only one process to execute here, other processes wait to exit

        final_data = pd.DataFrame()
        owd = os.getcwd()
        print('programming is finishing', owd)
        xl = pd.ExcelFile('2018_final_dataset_V_MPI.xlsx')
        data_2018 = xl.parse('Sheet1')
        x2 = pd.ExcelFile('2019_final_dataset_V_MPI.xlsx')
        data_2019 = x2.parse('Sheet1')
        x3 = pd.ExcelFile('2020_final_dataset_V_MPI.xlsx')
        data_2020 = x3.parse('Sheet1')
        final_data = pd.concat([data_2018, data_2019, data_2020], axis=0)  # concat both years
        final_data.to_excel('final_dataset_V_MPI.xlsx', index=False)

        print('--------CREATING FINAL DATABASE DONE------------')

        print("--- %s seconds ---" % (time.time() - start_time))


