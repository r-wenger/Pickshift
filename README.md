# Pickshift

## Cite this Script

If you use this script for academic research or as part of any scientific publication, we ask you to cite it. Below is a sample citation:

**APA Style**

```text
Author1, Author2, & Author3. (Year). PickShift: A Python Script for Geospatial Analysis Using Monte Carlo Simulations [Computer software]. Repository URL.
```

**MLA Style**

```text
Author1, Author2, and Author3. "PickShift: A Python Script for Geospatial Analysis Using Monte Carlo Simulations." Year, Repository URL.
```

**BibTeX Style**

```bibtex
@misc{PickShift,
  title={PickShift: A Python Script for Geospatial Analysis Using Monte Carlo Simulations},
  author={Author1 and Author2 and Author3},
  year={Year},
  url={Repository URL}
}
```

## Overview

The `pickshift_main` script performs geospatial analysis using Monte Carlo simulations to evaluate and quantify spatial errors. The script uses various Python libraries such as `pandas`, `geopandas`, `gdal`, and others for data manipulation, transformation, and spatial analysis.

## Dependencies

### Manual Installation

To run this script, you'll need the following Python packages:

- pandas
- geopandas
- numpy
- fiona
- osgeo
- shapely
- rasterstats
- os
- argparse
- configparser
- time
- tqdm

You can install these dependencies using pip:

```bash
pip install pandas geopandas numpy fiona osgeo shapely rasterstats argparse configparser tqdm
```

### Using Anaconda Environment

Alternatively, you can set up the environment using Anaconda with the provided `environment.yml` file. Run the following command to create a new environment:

```bash
conda env create -f environment.yml
```

To activate the environment, use:

```bash
conda activate <env_name>
```

Replace `<env_name>` with the name of the environment specified in `environment.yml`.

## Configuration File

The script requires a configuration file in INI format for its parameters. Here's a sample configuration file:

```ini
[DEFAULT]
GCP_txt = path/to/GCP.txt
extent_txt = path/to/extent.txt
poly_A_multi_txt = path/to/poly_A_multi.txt
buffer_txt = 0.5
crs_txt = 4326
runs_txt = 1000
resol_x_txt = 1.0
resol_y_txt = 1.0
outputf_txt = path/to/output/folder/
```

## How to Run

1. Ensure that you have all the required dependencies installed.
2. Place your configuration file in a location accessible by the script.
3. Run the script via the command line as follows:

```bash
python script_name.py -c path/to/config.ini
```

Replace `script_name.py` with the name of this script file and `path/to/config.ini` with the path to your configuration file.

## Functions

### `validate_config(config)`

Validates the configuration parameters. It checks whether the files exist and if the folder for output exists or not.

### `pickshift_main(config_file)`

The main function that performs all the tasks including:

- Reading the configuration
- Reading and preprocessing input files
- Spatial transformations
- Monte Carlo simulations
- Writing the results to output files

## Output

The script generates various output files:

1. Interpolated rasters of spatial errors.
2. Geopackages containing polygons with simulated Monte Carlo surface areas.
3. CSV files containing simulated points and polygons.

All output files are stored in the output folder specified in the configuration file.

## Performance Metrics

The script outputs statistics such as mean and standard deviation for different types of errors, both at the point and polygon level.

## Execution Time

The script prints the total running time at the end of the execution.

## Note

The code is quite extensive and performs multiple operations. Make sure to validate your input files and configurations before running it.