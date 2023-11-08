import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import fiona
from fiona import crs
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import shape, mapping
import rasterstats as rs
import os
import numpy.ma as ma
import rasterio
from rasterio.plot import plotting_extent
from rasterio.plot import show
import csv
import random
from random import *
from tqdm import tqdm
import math
import sys

import argparse
import configparser

import time
from time import monotonic

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def validate_config(config):
    # Check if files exist
    files_to_check = ['GCP', 'extent', 'polygons']
    for file_key in files_to_check:
        if not os.path.exists(config.get('DEFAULT', file_key)):
            raise FileNotFoundError(f"The file {config.get('DEFAULT', file_key)} does not exist!")

    # Check if the output folder exists. If not, try to create it
    output_folder = config.get('DEFAULT', 'outputf')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder at {output_folder}")

    try:
        float(config.get('DEFAULT', 'buffer'))
        int(config.get('DEFAULT', 'crs'))
        int(config.get('DEFAULT', 'runs'))
        float(config.get('DEFAULT', 'resol_x'))
        float(config.get('DEFAULT', 'resol_y'))
        float(config.get('DEFAULT', 'digit_error'))
        str(config.get('DEFAULT', 'douglas_peucker'))
        float(config.get('DEFAULT', 'tolerance'))
        str(config.get('DEFAULT', 'outputCSV_point_sim'))
        str(config.get('DEFAULT', 'outputCSV_poly_sim'))
        str(config.get('DEFAULT', 'outputSVE'))
    except ValueError:
        raise ValueError("One of the parameters in the config file is not correctly formatted.")


def pickshift_main(config_file):
  print("[INFO] Reading and validating configuration...")

  config = configparser.ConfigParser()
  config.read(config_file)

  validate_config(config)
  
  # Extract values from the config file
  GCP_txt = config.get('DEFAULT', 'GCP')
  extent_txt = config.get('DEFAULT', 'extent')
  poly_A_multi_txt = config.get('DEFAULT', 'polygons')
  buffer_txt = config.get('DEFAULT', 'buffer')
  crs_txt = config.get('DEFAULT', 'crs')
  runs_txt = config.get('DEFAULT', 'runs')
  resol_x_txt = config.get('DEFAULT', 'resol_x')
  resol_y_txt = config.get('DEFAULT', 'resol_y')
  digit_error = config.get('DEFAULT', 'digit_error')
  douglas_peucker = config.get('DEFAULT', 'douglas_peucker')
  tolerance = config.get('DEFAULT', 'tolerance')
  outputf_txt = config.get('DEFAULT', 'outputf')
  outputCSV_point_sim = config.get('DEFAULT', 'outputCSV_point_sim')
  outputCSV_poly_sim = config.get('DEFAULT', 'outputCSV_poly_sim')
  outputIDW = config.get('DEFAULT', 'outputSVE')
  outputGCPb = config.get('DEFAULT', 'outputGCPb')

  # In[ ]:
  print("[INFO] Loading spatial and tabular data...")
  GCP_df=pd.read_table(GCP_txt)
  emprise =  gpd.read_file(extent_txt)
  poly_A_multi =  gpd.read_file(poly_A_multi_txt)
  buffer=float(buffer_txt)
  crs=int(crs_txt)
  runs=int(runs_txt)
  digit_error = float(digit_error)
  douglas_peucker = str(douglas_peucker)
  tolerance = float(tolerance)
  resol_x=float(resol_x_txt)
  resol_y=float(resol_y_txt)

  if outputCSV_point_sim == 'True':
    outputCSV_point_sim = True
  else:
    outputCSV_point_sim = False

  if outputCSV_poly_sim == 'True':
    outputCSV_poly_sim = True
  else:
    outputCSV_poly_sim = False

  if outputIDW == 'True':
    outputIDW = True
  else:
    outputIDW = False

  if outputGCPb == 'True':
    outputGCPb = True
  else:
    outputGCPb = False

  if douglas_peucker == 'True':
    douglas_peucker = True
  else:
    douglas_peucker = False

  #Starting time
  start = monotonic() 


  # In[ ]:

  X_ref=GCP_df['Xref']
  Y_ref=GCP_df['Yref']
  X_ini=GCP_df['Xinit']
  Y_ini=GCP_df['Yinit']


  # In[ ]:


  #Calculating planimetric biases
  print("[INFO] Calculating planimetric biases from GCP file...")
  GCP_biasXY= abs(((X_ref-X_ini)**2 + (Y_ref-Y_ini)**2)**0.5)
  GCP_biasX= abs(X_ref-X_ini)
  GCP_biasY= abs(Y_ref-Y_ini)
  GCP_df.insert(4, "BiasXY", GCP_biasXY, allow_duplicates=False)
  GCP_df.insert(5, "BiasX", GCP_biasX, allow_duplicates=False)
  GCP_df.insert(6, "BiasY", GCP_biasY, allow_duplicates=False)


  # In[ ]:


  #Calculating biases and standard deviation
  mean_XY = GCP_df['BiasXY'].mean()
  std_XY= GCP_df['BiasXY'].std()
  mean_X = GCP_df['BiasX'].mean()
  std_X= GCP_df['BiasX'].std()
  mean_Y = GCP_df['BiasY'].mean()
  std_Y= GCP_df['BiasY'].std()
  print("[INFO] XY : mean planimetric bias = ",mean_XY, "std =", std_XY )
  print("[INFO] X  : mean planimetric bias = ",mean_X, "std =", std_X )
  print("[INFO] Y  : mean planimetric bias = ",mean_Y, "std =", std_Y )


  # In[ ]:


  #Converting GCP into spatial file
  point_gdf=gpd.GeoDataFrame(GCP_df,geometry=gpd.points_from_xy(GCP_df.Xinit,GCP_df.Yinit),crs=crs)


  # In[ ]:


  #Exporting in SHP point file
  outfp = "GCP_bias.gpkg"
  point_gdf.to_file(os.path.join(outputf_txt, outfp),crs=crs)
  if outputGCPb:
    print("[OUTPUT] GCP_bias.gpkg file created.")


  # In[ ]:


  #Creating width and height parameters
  xmin, ymin, xmax, ymax= emprise.total_bounds
  ht=abs((ymax-ymin)/resol_y)
  lg=abs((xmax-xmin)/resol_x)


  # In[ ]:


  #Interpolate biases
  print("[INFO] Interpolating biases over the extent to produce SVE map...")
  idwXY = gdal.Grid(os.path.join(outputf_txt, "SVE_XY.tif"), os.path.join(outputf_txt, "GCP_bias.gpkg"),zfield="BiasXY",
                  algorithm = "invdist:power=2:smoothing=1.0",
                  outputBounds = [xmin,ymax,xmax,ymin],
                  width=lg,height=ht)
  print("[OUTPUT] SVE_XY.tif file created.")
  idwXY=None
  idwX = gdal.Grid(os.path.join(outputf_txt, "SVE_X.tif"), os.path.join(outputf_txt, "GCP_bias.gpkg"),zfield="BiasX",
                  algorithm = "invdist:power=2:smoothing=1.0",
                  outputBounds = [xmin,ymax,xmax,ymin],              
                  width=lg,height=ht)
  print("[OUTPUT] SVE_X.tif file created.")
  idwX=None
  idwY = gdal.Grid(os.path.join(outputf_txt, "SVE_Y.tif"), os.path.join(outputf_txt, "GCP_bias.gpkg"),zfield="BiasY",
                  algorithm = "invdist:power=2:smoothing=1.0",
                  outputBounds = [xmin,ymax,xmax,ymin],
                  width=lg,height=ht)
  print("[OUTPUT] SVE_Y.tif file created.")
  idwY=None


  # In[ ]:


  #Export rasters
  ESV_data_XY = rasterio.open(os.path.join(outputf_txt, 'SVE_XY.tif')).read(1, masked=True)
  ESV_data_X = rasterio.open(os.path.join(outputf_txt, 'SVE_X.tif')).read(1, masked=True)
  ESV_data_Y = rasterio.open(os.path.join(outputf_txt, 'SVE_Y.tif')).read(1, masked=True)

  ESV_data_XY_transform = rasterio.open(os.path.join(outputf_txt, 'SVE_XY.tif')).transform
  ESV_data_X_transform = rasterio.open(os.path.join(outputf_txt, 'SVE_X.tif')).transform
  ESV_data_Y_transform = rasterio.open(os.path.join(outputf_txt, 'SVE_Y.tif')).transform

  # In[ ]:
  
  
    #Converting to polygons
  poly_A= poly_A_multi.explode(index_parts=True)
  

#Pixel mean for SVE XY
  mean_pixel_error = np.mean(ESV_data_XY)

  average_distances_per_polygon = []

  for _, poly in poly_A.iterrows():
      nodes_of_poly = [Point(pt) for pt in np.asarray(poly['geometry'].exterior.coords)]
      
      distances = []
      for i in range(1, len(nodes_of_poly)):
          dist = nodes_of_poly[i-1].distance(nodes_of_poly[i])
          distances.append(dist)
      average_distances_per_polygon.append(np.mean(distances))

  final_average_distance = np.mean(average_distances_per_polygon)

  #Calculating the ratio
  ratio = mean_pixel_error / final_average_distance
  
  print('[INFO] The ratio between mean SVE and mean nodes distance is ' + str(ratio))
  #If the ratio is higher or equal to one and Douglas-Peucker set to 'False'
  if (ratio >= 1) and (douglas_peucker == False):
    response = str(input('[WARNING] Monte-Carlo simulations will probably result in topological errors. Do you still want to continue without Douglas-Peucker simplification? (y/n)'))

    if response != 'y':
      print('[INFO] Shutting down. We suggest using Douglas-Peucker with a tolerance of ' + str(int(final_average_distance*2)) + ". Increase this value if needed. Please edit the configuration file (config.conf).")
      sys.exit()



# In[ ]:


  #Summit extraction
  col = poly_A.columns.tolist()[0:2]
  nodes = gpd.GeoDataFrame(columns=col)
  total_rows = len(poly_A)

  print("[INFO] Assignment of mean and std error to each polygon nodes according to the buffer size...")
  for index, row in tqdm(poly_A.iterrows(), total=poly_A.shape[0], desc=""):
    for pt in np.asarray(row['geometry'].exterior.coords):
      nodes = pd.concat([nodes, gpd.GeoDataFrame({'id': [int(row['id'])], 'geometry': [Point(pt)]})], ignore_index=True)

  nodes.set_geometry('geometry')

  

  # In[ ]:

  nodes.drop_duplicates(keep='first',inplace=True)

  # In[ ]:

  schema = {
    'geometry': 'Polygon',
    'properties': {
        'id': 'str'
    }
  }
  #crs_fiona = crs.from_epsg(crs)

  #Calculating buffer + export
  pts_to_poly = nodes.copy()
  nodes = nodes.set_geometry('geometry')
  pts_to_poly["geometry"] = nodes.geometry.buffer(buffer)
  pts_to_poly = pts_to_poly.set_geometry('geometry')
  
  pts_to_poly['id'] = pts_to_poly['id'].astype(int)
  pts_to_poly['type'] = pts_to_poly['type'].astype(str)
  pts_to_poly = pts_to_poly.drop(columns=['type'])

  output_path = os.path.join(outputf_txt, "Buffer.gpkg")
  pts_to_poly.to_file(output_path, crs=crs, schema=schema, driver="GPKG")


  # In[ ]:


  #Extracting SVE in X and convert it into geodataframe
  ESV_X = rs.zonal_stats(output_path,
                         ESV_data_X,
                         nodata=-999,
                         affine=ESV_data_X_transform,
                         geojson_out=True,
                         copy_properties=True,
                         stats="mean std")

  ESV_dfX = gpd.GeoDataFrame.from_features(ESV_X)
  ESV_dfX=ESV_dfX.rename(columns={'mean':'Xmean','std':'Xstd'})


  # In[ ]:


  #Extracting SVE in Y and convert it into grodataframe
  ESV_Y = rs.zonal_stats(ESV_dfX,
                         ESV_data_Y,
                         nodata=-999,
                         affine=ESV_data_Y_transform,
                         geojson_out=True,
                         copy_properties=True,
                         stats="mean std")
  ESV_dfY = gpd.GeoDataFrame.from_features(ESV_Y)
  ESV_dfY=ESV_dfY.rename(columns={'mean':'Ymean','std':'Ystd'})


  # In[ ]:


  #Extracting SVE in XY and convert it into grodataframe
  ESV_XY = rs.zonal_stats(ESV_dfY,
                         ESV_data_XY,
                         nodata=-999,
                         affine=ESV_data_XY_transform,
                         geojson_out=True,
                         copy_properties=True,
                         stats="mean std")
  ESV_dfXY = gpd.GeoDataFrame.from_features(ESV_XY)
  ESV_dfXY=ESV_dfXY.rename(columns={'mean':'XYmean','std':'XYstd'})


  # In[ ]:


  #Convert into points
  ESV_XYXY=ESV_dfXY.copy()
  ESV_XYXY['geometry']=ESV_XYXY['geometry'].centroid


  # In[ ]:



  meanX = ESV_XYXY.Xmean
  stdX = ESV_XYXY.Xstd
  meanY = ESV_XYXY.Ymean
  stdY = ESV_XYXY.Ystd
  meanXY = ESV_XYXY.XYmean
  stdXY = ESV_XYXY.XYstd
  id = ESV_XYXY.id
  num_reps = len(ESV_XYXY)
  X0 = ESV_XYXY.geometry.x
  Y0 = ESV_XYXY.geometry.y
  erreur = digit_error
  prob = [-1, 1]
  res_dfi = []

  # Create dictionaries to store betax1 and betay1 values for each polygon
  dict_betax1 = {}
  dict_betay1 = {}

  for i in tqdm(range(runs), desc="[INFO] Monte Carlo translations"):
      Xgaus = np.random.normal(meanX, stdX, num_reps)
      Ygaus = np.random.normal(meanY, stdY, num_reps)
      XYgaus = np.random.normal(meanXY, stdXY, num_reps)

      df = pd.DataFrame(index=range(num_reps), data={'id': id,
                                                     'X': X0,
                                                     'Y': Y0,
                                                     'Xgaus': Xgaus,
                                                     'Ygaus': Ygaus,
                                                     'XYgaus': XYgaus,
                                                     'Error': erreur})

      for unique_id in df['id'].unique():
          # betax1 identical for all nodes in the same polygon
          betax1 = np.random.choice(prob, size=1, replace=True, p=[0.5, 0.5])[0]
          dict_betax1[unique_id] = betax1

          # betay1 varies between nodes of the same polygon
          betay1_values = np.random.choice(prob, size=df[df['id'] == unique_id].shape[0], replace=True, p=[0.5, 0.5]).tolist()
          dict_betay1[unique_id] = betay1_values

      # Apply betax1 and betay1 values
      df['Beta1_X'] = df['id'].map(dict_betax1)
      df['Beta1_Y'] = df['id'].apply(lambda x: dict_betay1[x].pop(0) if dict_betay1[x] else np.nan)

      # betax2 and betay2 are drawn randomly for each node
      df['Beta2_X'] = np.random.choice(prob, size=num_reps, replace=True, p=[0.5, 0.5])
      df['Beta2_Y'] = np.random.choice(prob, size=num_reps, replace=True, p=[0.5, 0.5])

      df['X1'] = df['X'] + df['Xgaus'] * df['Beta1_X'] + (df['Error'] * df['Beta2_X'])
      df['Y1'] = df['Y'] + df['Ygaus'] * df['Beta1_Y'] + (df['Error'] * df['Beta2_Y'])

      for j in range(0, num_reps):
          res_dfi.append([i, df['id'][j], df['Xgaus'][j], df['Ygaus'][j], df['XYgaus'][j], df['X1'][j], df['Y1'][j]])

  results_df = pd.DataFrame.from_records(res_dfi, columns=['run', 'id','ESV_X','ESV_Y','ESV_XY','X','Y'])                                            
  gdf = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.X, results_df.Y), crs=crs)

  # In[ ]:

  #Convert into polygons and surface calculation
  def aggregate_to_polygon(group):
      if douglas_peucker == True:
        return Polygon(group['geometry'].values).simplify(tolerance, preserve_topology=False)
      else:
        return Polygon(group['geometry'].values)

  df2 = gdf.groupby(['run','id']).apply(aggregate_to_polygon).reset_index()
  df2 = df2.rename(columns={0: 'geometry'})

  df2['id'] = pd.to_numeric(df2['id'])
  df2.sort_values(by=['id'], ascending=True,
                            inplace=True, ignore_index=True)

  geodf = gpd.GeoDataFrame(df2, geometry='geometry')
  geodf['Area']=geodf.area

  # In[ ]:

  grouppoly=geodf.groupby(["id"])[["Area"]].describe()
  grouppoly.columns = grouppoly.columns.droplevel(0)
  for n in tqdm(range(len(grouppoly)), desc="[INFO] Calculating surface uncertainties for each polygon"):
    grouppoly['initial area']=list(poly_A.area)
    grouppoly['total uncertainty']=((0.5*(grouppoly['max']-grouppoly['min']))/grouppoly['mean'])*100
    q2=np.asarray(geodf.groupby(["id"])[["Area"]].quantile(0.025))
    q9=np.asarray(geodf.groupby(["id"])[["Area"]].quantile(0.975))
    grouppoly['conf_interv']=q9-q2
    grouppoly['95% uncertainty']= ((0.5*grouppoly['conf_interv'])/grouppoly['mean'])*100 
  #grouppoly

  end = monotonic() 
  print("[INFO] Running time : ", str(round(end-start, 2)), "seconds")

  #Jointure et export
  poly_MC = poly_A_multi.merge(grouppoly, on='id')
  poly_MC.to_file(os.path.join(outputf_txt, "poly_MC.gpkg"), driver="GPKG")
  print("[OUTPUT] poly_MC.gpkg file created.")

  # Define the paths of the files
  file1 = os.path.join(outputf_txt, 'poly_sim_MC.csv')
  file2 = os.path.join(outputf_txt, 'point_sim_MC.csv')

  # Check if the files exist and remove them
  for file in [file1, file2]:
    if os.path.exists(file):
      os.remove(file)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #remove geometry column
    if outputCSV_poly_sim is True:
      geodf.set_geometry('geometry', inplace=True)
      geodf.to_csv(os.path.join(outputf_txt, 'poly_sim_MC.csv'), index=False)
      print("[OUTPUT] poly_sim_MC.csv file created.")

    if outputCSV_point_sim is True:
      gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt if x is not None else None)
      gdf = gdf.drop(columns=['geometry'])
      gdf.to_csv(os.path.join(outputf_txt, 'point_sim_MC.csv'), index=False)
      print("[OUTPUT] point_sim_MC.csv file created.")


    if outputIDW is False:
      os.remove(os.path.join(outputf_txt, 'SVE_XY.tif'))
      os.remove(os.path.join(outputf_txt, 'SVE_X.tif'))
      os.remove(os.path.join(outputf_txt, 'SVE_Y.tif'))

    if outputGCPb is False:
      os.remove(os.path.join(outputf_txt, outfp))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PickShift script for geospatial analysis.")
  parser.add_argument("-c", "--config", help="Path to the configuration file.", required=True)
  
  args = parser.parse_args()
  
  pickshift_main(args.config)
