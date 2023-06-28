import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.enums import Resampling

import geopandas as gpd
import os
import glob
import math
import shapely
from rasterio import features
from shapely.geometry import Polygon, LineString, Point, shape
from osgeo import gdal, ogr
import pandas as pd
import numpy as np
from pathlib import Path
import json
from shapely.geometry import box
from fiona.crs import from_epsg
import pycrs

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def getFeatures(gdf):
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def moving_window_images(minx, maxx, miny, maxy, image_no, resolution, mask, satellite, directory):
    # w_px = math.ceil((max_x - min_x) / resolution)
    # h_px = math.ceil((max_y - min_y) / resolution)
    # shape_window = w_px, h_px
    # transform = rasterio.transform.from_bounds(*bounds, w_px, h_px)

    bbox = box(minx, miny, maxx, maxy)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=32630)
    coords = getFeatures(geo)
    # geo.to_file("bbox2.shp")

    with rasterio.open(mask, 'r') as src:
        output_labelled = f'./{directory}/mask/{image_no}.tif'
        output_satellite = f'./{directory}/satellite/{image_no}.tif'
        out_labels_meta = src.meta.copy()
        out_sat_meta = src.meta.copy()

        # crop label image
        out_labels, out_transform_labels = mask(src,shapes=coords,crop=True)

        # crop satellite image
        with rasterio.open(satellite, 'r') as sat:
            out_sat, out_transform_sat = mask(sat,shapes=coords,crop=True)

        # check that label image contains cocoa or non-cocoa value (1,2)
        max = np.max(out_labels)
        if max>0:
        #update metadata
            out_labels_meta.update({"driver": "GTiff", "height": out_labels.shape[1],  
            "width": out_labels.shape[2], "transform": out_transform_labels})
            out_sat_meta.update({"driver": "GTiff", "height": out_sat.shape[1],  
            "width": out_sat.shape[2], "transform": out_transform_sat,  "count": n_bands})

            #write to file
            with rasterio.open(output_labelled, 'w', **out_labels_meta) as dst:
                dst.write(out_labels)
            with rasterio.open(output_satellite, 'w', **out_sat_meta) as dst:
                dst.write(out_sat)
        
def save_window(x_ind, y_ind, window_size, image_no, mask, satellite, n_bands, directory):
    output_mask = f'./{directory}/mask/{image_no}.tif'
    output_satellite = f'./{directory}/satellite/{image_no}.tif'
    output_inter_mask = f'./{directory}/inter_mask/{image_no}.tif'
    output_inter_satellite = f'./{directory}/inter_satellite/{image_no}.tif'
    window=Window(x_ind, y_ind, window_size, window_size)

    # get window transforms
    with rasterio.open(mask) as file:
        mask_local = file.read(window=window)
        transform_mask = file.window_transform(window)

    with rasterio.open(satellite) as file:
        sat_local = file.read(window=window)
        transform_sat = file.window_transform(window)
    
    max = np.max(mask_local)
    count_1 = np.count_nonzero(mask_local == 1 )
    count_2 = np.count_nonzero(mask_local == 2 )
    count_3 = np.count_nonzero(mask_local == 3 )

    # set aside intercrop only areas
    if max == 3 and count_1 == 0 and count_2 == 0:
        with rasterio.open(output_inter_mask, 'w',
        driver='GTiff', width=window_size, height=window_size, count=1,
        dtype=rasterio.uint8, transform=transform_mask) as dst:
            dst.write(mask_local)

        # Save clip of satellite
        with rasterio.open(output_inter_satellite, 'w',
        driver='GTiff', width=window_size, height=window_size, count=n_bands,
        dtype=rasterio.rasterio.float32, transform=transform_sat) as dst:
            dst.write(sat_local)
        return True
    
    if count_1 >= window_size*window_size*0.10 or count_2 >= window_size*window_size*0.10:
        with rasterio.open(output_mask, 'w',
        driver='GTiff', width=window_size, height=window_size, count=1,
        dtype=rasterio.uint8, transform=transform_mask) as dst:
            dst.write(mask_local)

        # Save clip of satellite
        with rasterio.open(output_satellite, 'w',
        driver='GTiff', width=window_size, height=window_size, count=n_bands,
        dtype=rasterio.rasterio.float32, transform=transform_sat) as dst:
            dst.write(sat_local)
        return True

def create_dataset(satellite, mask, directory, n_bands):
    print('Creating training dataset...')
    image = 0

    open_mask = rasterio.open(mask)
    mask_array = open_mask.read()
    open_satellite = rasterio.open(satellite)
    sat_array = open_satellite.read()

    # sampling size
    labelled_cacao = np.count_nonzero(mask_array == 1)
    labelled_forest = np.count_nonzero(mask_array == 2)
    total_labelled = labelled_cacao+labelled_forest
    percent_cacao=labelled_cacao/total_labelled
    print('total labelled = ', total_labelled)
    print('percent cacao = ', percent_cacao)
    test_total = total_labelled*0.10
    print('test minimum pixels = ', test_total)
    test_cacao = labelled_cacao*0.10
    test_forest = labelled_cacao*0.10

    # set extent to tile (clipped combined bands)
    with open_satellite as src:
        profile = src.profile
        transform = src.transform
        width = src.width
        height = src.height

    resolution = 10
    # compute tile size 
    w_px = width
    h_px = height
    shape_tile = w_px, h_px

    # initialize moving window
    window_size = 128
    x_ind = 0
    y_ind = 0
    stride = 128
    print('window initialized')


    #traverse 
    while x_ind < (mask_array.shape[1]-window_size-1):
        # save image
        while y_ind < (mask_array.shape[2]-window_size-1):
            # save image
            image+=1
            # print(f'window {image}:{x_ind},{y_ind}')
            # move down
            y_ind+=stride
            save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)
        save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)
        image+=1
        # print(f'window {image}:{x_ind},{y_ind}')
        # move across
        x_ind+=stride
        y_ind=0
        save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)
    
    while y_ind < (mask_array.shape[2]-window_size-1):
            # save image
            image+=1
            # print(f'window {image}:{x_ind},{y_ind}')
            # move down
            y_ind+=stride
            save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)

    print('Finished creating training dataset!!!')

def create_grid(satellite, mask):
    print('Creating grid...')
    image = 0

    open_mask = rasterio.open(mask)
    mask_array = open_mask.read()
    open_satellite = rasterio.open(satellite)
    sat_array = open_satellite.read()

    # set extent to tile (clipped combined bands)
    with open_satellite as src:
        profile = src.profile
        transform = src.transform
        width = src.width
        height = src.height
        bbox = src.bounds

    resolution = 10
    # compute tile size 
    w_px = width
    h_px = height
    shape_tile = w_px, h_px

    # initialize centroid at top left
    min_x = bbox[0] # 699 960
    max_x = bbox[2] # 809 660
    min_y = bbox[1] # 590 220
    max_y = bbox[3] # 700 020
    upper_y = min_y + 1500 # 590 370
    upper_x = max_x - 1500 # 809 510
    centroid_x = min_x + 1500 # 700 110
    centroid_y = max_y - 1500 # 699 870
    grid_x = []
    grid_y = []
    stride = 3000

    grid_x.append(centroid_x)
    grid_y.append(centroid_y)

    #traverse 
    while centroid_x < upper_x:
        # save image
        while centroid_y  > upper_y:
            # move down
            centroid_y -=stride
            if centroid_y > upper_y:
                grid_x.append(centroid_x)
                grid_y.append(centroid_y)
        if centroid_y > upper_y:
            grid_x.append(centroid_x)
            grid_y.append(centroid_y)
        # move across
        centroid_x += stride
        centroid_y = max_y-1500.0
        if centroid_x < upper_x:
            grid_x.append(centroid_x)
            grid_y.append(centroid_y)

    
    points_df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_x, grid_y), crs="EPSG:32630")
    points_df.to_file('grid.shp')
    
    print('Finished creating grid!!!')
    return grid_x, grid_y, points_df

def dataset_from_grid(satellite_file, mask_output, directory, n_bands, size, grid):
    print('Creating dataset from grid...')
    image = 0

    open_mask = rasterio.open(mask_output)
    mask_array = open_mask.read()
    open_satellite = rasterio.open(satellite_file)
    sat_array = open_satellite.read()

    points_df = grid[2]
    sample = 0
    for x, y in zip(grid[0], grid[1]):
        # if sample < 100:
        # get raster index 
        x_ind, y_ind = open_satellite.index(x, y)
        # print(x_ind, y_ind)
        x_ind, y_ind = x_ind-size//2, y_ind-size//2
        # print(x_ind, y_ind)
        result = save_window(x_ind, y_ind, size, image, mask_output, satellite_file, n_bands, directory)
        image+=1
        if result == True:
            sample+=1

    print('Finished creating dataset from grid!!!')

def mask_rasters(satellite_file, polygons_file, mask_output):
    print('Creating mask...')
    df = gpd.read_file(polygons_file) # contains labelled ground truth polygons
    
    # set extent to satellite tile for all (align rasters)
    with rasterio.open(satellite_file, 'r') as src:
        profile = src.profile
        out_shape = src.width, src.height
        transform = src.transform

    # # create mask => data only on labelled pixels
    labelled_shapes = []
    for shape in df['geometry']:
        labelled_shapes.append(shape)

    # transform = rasterio.transform.from_bounds(*bounds, w_px, h_px)
    geom_value = [(shape, cost) for shape, cost in zip(df['geometry'], df['value'])]

    # rasterize ground truth polygons using satellite as extent
    output_raster = rasterize(
       geom_value,
       out_shape=out_shape,
       all_touched=False, 
       transform=transform,
       dtype=rasterio.int32)

    with rasterio.open(mask_output, 'w',
        driver='GTiff',
        dtype=rasterio.int32,
        count=1,
        width=src.width,
        height=src.height,
        transform=src.transform
        ) as dst:
            dst.write(output_raster, indexes=1)
    
    print('Mask is ready!!!')

if __name__ == "__main__":
    
    satellite_file = './data/virtual-raster/dec_MSI_temp_SAR_vv_ideal.tif'
    polygons_file = './data/polygon-layers/mono_inter_non_cocoa.shp'
    mask_output = 'mono_inter_non_cocoa.tif' # name to use for created mask raster
    directory = './data/dec_MSI_temp_SAR_vv_cloudless'     # created large mask raster

    # to create label raster
    mask_rasters(satellite_file, polygons_file, mask_output)

    # set parameters
    n_bands = 10
    size = 128

    # grid = create_grid(satellite_file, mask_output)  
    scale_factor = 8

    # # create training dataset (along grid)
    # dataset_from_grid(satellite_file, mask_output, directory, n_bands, size, grid)

    # # create training dataset (cover entire raster)
    create_dataset(satellite_file, mask_output, directory, n_bands)
    
    
    '''POSSIBLY USEFUL'''
    # resample rasters
    # for raster_file in raster_files:
    #     resampled_raster = raster_file[:-4]+'_resampled.tif'
    #     with rasterio.open(reproj_directory + '/'+ raster_file, 'r') as dataset:
    #         data = dataset.read(1)
    #         # Get the metadata of the source raster
    #         profile = dataset.profile
            
    #         # Calculate the new resolution
    #         new_resolution = 10
            
    #         # Calculate the new transform based on the desired resolution
    #         new_transform = rasterio.transform.from_origin(profile['transform'][2], profile['transform'][5], new_resolution, new_resolution)
            
    #         # Create a new raster file for the resampled data
    #         with rasterio.open(reproj_directory + '/'+ resampled_raster, 'w', **profile) as dst:
    #             # Perform the resampling
    #             dst.write(data, indexes=1)
    #             dst.transform = new_transform

    # reproject many rasters
    reproj_directory = 'temp_raster_stack/resample'
    dst_crs = 'EPSG:32630'
    raster_files = os.listdir(reproj_directory)
    # for raster_file in raster_files:
    #     reproj_raster = raster_file[:-4]+'_reprojected.tif'
    #     # Reproject raster
    #     print(f'Reprojecting raster: {raster_file}')
    #     with rasterio.open(reproj_directory + '/'+ raster_file, 'r') as src:
    #         transform, width, height = rasterio.warp.calculate_default_transform(
    #         src.crs, dst_crs, src.width, src.height, *src.bounds)
    #         kwargs = src.meta.copy()
    #         kwargs.update({
    #             'crs': dst_crs,
    #             'transform': transform,
    #             'width': width,
    #             'height': height
    #         })
    #         data = src.read()

    #         with rasterio.open(reproj_directory + '/'+ reproj_raster, 'w', **kwargs) as dst:
    #             for i in range(1, src.count + 1):
    #                 # reproject raster(s) to config CRS
    #                 rasterio.warp.reproject(source=rasterio.band(src, i),
    #                     destination=rasterio.band(dst, i),
    #                     src_transform=src.transform,
    #                     src_crs=src.crs,
    #                     dst_transform=transform,
    #                     dst_crs=dst_crs,
    #                     resampling=Resampling.bilinear)
