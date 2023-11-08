## PickShift

### Cite this software

If you use PickShift for academic research or as part of any scientific publication, we ask you to cite it. Below is a sample citation:

### Overview

PickShift performs geospatial analysis using Monte Carlo simulations to (1) compute the spatially variable error affecting any historical planimetric data and (2) quantify the surficial uncertainty associated to digitized features.
The script uses various Python libraries such as `pandas`, `geopandas`, `gdal`, and others for data manipulation, transformation, and spatial analysis.

PickShift consists of three files:
- environment.xml <br>
It allows the operator to automatically configure and build an Anaconda environment that suits the appropriate operating system.
- config.conf <br>
It contains the location of the required inputs and few parameters that can be modified by the operator.
- pickshift.py <br>
It corresponds to the main source code that has to be run once the Anaconda environment is created and the configuration file parameterized.

Here is a link to download those three files:
[link]

### Video tutorial

This video explains how to use PickShift. We recommend watching it.
[link]

### Test data

Here is a link to download the data presented in our article:
[link]

## Installation

We recommend using Anaconda to run PickShift, as it allows to automatically configure and build an Anaconda environment that suits the appropriate operating system. 
Please refer to the official instructions to install it, depending on your operating system: https://docs.anaconda.com/free/anaconda/install/index.html

Once Anaconda is installed on your operating system and the Anaconda environement created, you will be able to run PickShift from the terminal (for Linux/macOS) or the Anaconda prompt (for Windows).

### Create PickShift environment

Set up the PickShift environment with the provided `environment.yml` file. 
Run the following command to create the PickShift environment:

`conda env create -f environment.yml`

Run the following command to activate the PickShift environment:

`conda activate envpickshift`

## Run PickShift

Once the PickShift environment is created and activated, you can run PickShift using the following command:

`python pickshift.py -c config.conf`

NB: The required inputs must have been prepared and the configuration file (config.conf) parameterized before running PickShift. See sections below.



## Inputs required

PickShift requires three inputs that have be to prepared by the operator using any GIS software (e.g. QGIS, ArcGIS): 
 - a shapefile of the spatial extent of the studied area (extent, geopackage format); 
 - a shapefile of the features digitized from the historical planimetric data (polygons, geopackage format); 
 - a set of independent ground control points (GCP, txt format).

The input files must share the same projected coordinates system.

| File name | File format | Description/Recommendations|
|-----------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GCP       | .txt        | Contains the coordinates (XY) of each pair of Ground Control Points, picked from the targeted and the reference planimetric data. The coordinates can be in any projected coordinate system, in meters. The file must be tab-separated and contain only the following fields: ‘Xref’, ‘Yref’, ‘’Xinit’, ‘Yinit’. |
| polygons  | .gpkg       | Corresponds to the features of interests digitized from the targeted planimetric data. The file must include the two following fields: <br>- ‘id’ (the automatic QGIS ‘fid’ field is not enough) <br>- type<br>                           |
| extent    | .gpkg       | Corresponds to the spatial extent of the studied area, where the spatially-variable error is interpolated. We suggest to delineate it a bit larger than the area covered by the polygons.                                                                                                                        |


## Configuration File

The script requires a configuration file (config.conf) for its parameters. Here's a screenshot of the configuration file:

![](config_file.png)

The following table describes how to fill it. 

| Parameter name      | Type      | Units  | Description                                                                              | Comments/Recommendations |
|---------------------|-----------|--------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| GCP                 | character | /      | Location of the GCP file.                                                                | /                                                                               |
| extent              | character | /      | Location of the extent file.                                                             | /                                                                               |
| polygons            | character | /      | Location of the polygons file.                                                           | /                                                                               |
| crs                 | integer   | /      | Projected coordinate system of the input files.                                          | EPSG format. Must be the same for all input files.                              |
| resol_x             | integer   | meters | X resolution of the SVE raster.                                                          | Use the same resolution as the targeted data resolution.                        |
| resol_y             | integer   | meters | Y resolution of the SVE raster.                                                          | Use the same resolution as the targeted data resolution.                        |
| buffer              | integer   | meters | Radius of the buffer used to extract the SVE around each vertices.                       | Ten times the resolution of the targeted data.                                  |
| runs                | integer   | /      | Number of Monte-Carlos simulations.                                                      | 1000 simulations is the right number, but we suggest to first test with 10.     |
| digit_error         | integer   | meters | Digitization error.                                                                      | Minimum the same value as the targeted data resolution.                         |
| douglas_peucker     | boolean   | /      | 'True’ : to simplify the polygons after reconstruction.'False’ : without simplification. | Use only if the Monte-Carlo simulations induces topological errors.             |
| tolerance           | integer   | meters | Tolerance value used to simplify the polygons. For more details, refer to the [geopandas documentation.](https://shapely.readthedocs.io/en/latest/manual.html#object.simplify)                   | We recommend to visually inspect the results and adjust the value if necessary. |
| outputf             | character | /      | Name of the output folder.                                                               | It is created if it doesn’t exist.                                              |
| outputGCPb          | boolean   | /      | Exports the punctual biases. ‘True’ or ‘False’                                           | /                                                                               |
| outputCSV_point_sim | boolean   | /      | Exports the translated vertices. ‘True’ or ‘False’                                       | This file may be large.                                                         |
| outputCSV_poly_sim  | boolean   | /      | Exports the reconstructed polygons. ‘True’ or ‘False’                                    | This file may be large.                                                         |
| output_SVE          | boolean   | /      | Exports the interpolated SVE rasters. ‘True’ or ‘False’                                  | This file may be large.                                                         |




## Outputs description

The following table lists the differents output files PickShift exports.

| File name    | File format | Description |
|--------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GCP_bias     | .gpkg       | Corresponds to the GCPs picked from the targeted planimetric data, to which the punctual planimetric bias values are joined.                                                                                                           |
| SVE_XY       | .tif        | Spatially-variable error in XY, i.e. spatial interpolation (IDW) of the punctual planimetric biases.                                                                                                                                   |
| SVE_X        | .tif        | Spatially-variable error in X, i.e. spatial interpolation (IDW) of the punctual planimetric biases.                                                                                                                                    |
| SVE_Y        | .tif        | Spatially-variable error in Y, i.e. spatial interpolation (IDW) of the punctual planimetric biases.                                                                                                                                    |
| Buffer       | .gpkg       | Shapefile of the buffers from which the values of SVE are extracted around each polygons vertices.                                                                                                                                     |
| point_sim_MC | .csv        | Translated vertices resulting from the Monte-Carlo simulations.                                                                                                                                                                        |
| poly_sim_MC  | .csv        | Reconstructed polygons from the translated vertices.                                                                                                                                                                                   |
| poly_MC      | .gpkg       | Aggregated results. Contains the initial polygons geometry, associated with the following statistics: mean surface, standard deviation, minimum, percentiles, maximum, total uncertainty (%), confidence interval, 95% uncertainty (%) |