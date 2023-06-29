This README.md file was generated on 2023-06-28 by Adele Therias according to the 4TU.ResearchData [Guidelines for creating a README file](https://data.4tu.nl/info/en/use/publish-cite/upload-your-data-in-our-data-repository) and the Cornell University template [Guide to writing "readme" style metadata](https://cornell.app.box.com/v/ReadmeTemplate).
**Latest update:** 2023-06-28

# Integrating radar and multi-spectral data to detect cocoa crops: a deep learning approach

## Table of Contents

- [1. GENERAL INFORMATION](#1-general-information)
- [2. METHODOLOGICAL INFORMATION](#2-methodological-information)
  - [2.1 Research questions, methods and envisioned uses](#21-research-questions-methods-and-envisioned-uses)
  - [2.2 Methods for processing the data](#22-methods-for-processing-the-data)
  - [2.3 Methods for machine learning](#23-methods-for-ml)
  - [2.4 Methods for analysis](#23-methods-for-analysis)
  - [2.5 Instrument- or software-specific information](#24-instrument--or-software-specific-information)
- [3. SHARING/ACCESS INFORMATION](#3-sharingaccess-information)
  - [3.1 Licenses/restrictions placed on the data](#31-licensesrestrictions-placed-on-the-data)
  - [3.2 Recommended citation for this dataset](#32-recommended-citation-for-this-dataset)
- [4. GETTING STARTED](#4-getting-started)


# 1. GENERAL INFORMATION

## 1.1 Title of Project
Integrating radar and multi-spectral data to detect cocoa crops: a deep learning approach

## 1.2 Description
This MSc Geomatics thesis aims to evaluate the impact of combining SAR and MSI data in the training of a CNN for cocoa detection, in order to demonstrate the importance of texture, moisture and canopy characteristics in identifying cocoa canopies. It was carried out between November 2022 and June 2023. The full text can be accessed [here](http://resolver.tudelft.nl/uuid:314e12c9-c3bc-478b-b664-d0c0680f3caf).

## 1.3 Author Information
A. Principal Investigator  
- Name: Adele Therias
- Institution: Delft University of Technology
- Address: Julianalaan 134, 2628BL Delft, South-Holland, The Netherlands
- Email: A.M.Therias@student.tudelft.nl

B. First supervisor
- Name: Dr. Azarakhsh Rafiee
- Institution: Delft University of Technology
- Address: Julianalaan 134, 2628BL Delft, South-Holland, The Netherlands
- Email: A.Rafiee@tudelft.nl

C. Second supervisor
- Name: Dr. Stef Lhermitte
- Institution: Delft University of Technology
- Address: Julianalaan 134, 2628BL Delft, South-Holland, The Netherlands
- Email: S.Lhermitte@tudelft.nl

D. Company supervisor
- Name: Philip van der Lugt
- Institution: Meridia Land B.V.

## 1.4 Keywords
agriculture, synthetic aperture radar, multi-spectral imagery, convolutional neural network

## 1.5 Language
English

# 2. METHODOLOGICAL INFORMATION
## 2.1 Research questions, methods and envisioned uses
This research aims to advance knowledge about the relevance of different open source datasets in order to improve the detection of cocoa crops using machine learning. For example, this work could be used to further the cocoa detection in industry for the enforcement of the European Union Deforestation Regulation.

**Research question:** To what extent can a CNN trained with multispectral and SAR datasets enable the automated detection of cocoa crops in Ghana? <br>
**Sub-questions**
- How does the combination of MSI and SAR data affect the results of cocoa parcel segmentation trained with data from a single day?
- How does the combination of MSI and SAR data affect the results of cocoa parcel segmentation trained with temporal datasets?
- Why does the use of different polarizations (i.e. Vertical-Vertical (VV) or Vertical-Horizontal (VH)) affect the influence of SAR datasets on the cocoa segmentation results?
- What is the impact of SAR and MSI training data on the detection of intercrop cocoa?

## 2.2 Methods for processing the data
All relevant functions in preprocessing.py

### 2.2.1 Satellite data
S1 and S2 datasets are downloaded via the [WEkEO](https://www.wekeo.eu/) JupyterHub (Earth Observation Server) using the [WEkEO Harmonized Data Access API](https://help.wekeo.eu/en/collections/3530725-wekeo-harmonized-data-access). The MSI datasets (Level 2A) are filtered to contain less than 15% cloud cover and efforts are made to select imagery distributed across the wet and dry seasons. The year 2020-2021 is selected because it contains datasets that match as closely as possible to an even distribution across the seasons, and because this timeline overlaps with the ground truth dataset collection dates. The SAR datasets (Level 1C) are filtered to IW mode and GRD products only. For the dry season stack, the images are selected as close as possible to the dates of MSI data.<br>
- Project MSI data to UTM zone 30N and re-sample all 20 m resolution bands to 10 m resolution using bi-linear interpolation
- Project SAR to UTM zone 30N, re-sample to 10 m resolution and clip to study area
- Stack MSI and SAR in various virtual rasters with each file saved as a separate band.

### 2.2.2 Polygon data
After collecting the ground truth polygons from Meridia, the following steps are conducted using QGIS:<br>
- Quality-control ground truth polygons: conduct visual checks on cocoa ground truth to check for possible misclassification, such as overlaps between cocoa and non-cocoa ground truth
- Cocoa polygons: remove any polygons classified as "sparse," ”unknown” or ”both” to keep only ”intercrop” and ”monocrop” cultivation types. Remove any intercrop polygons intersecting with monocrop polygons and vice-versa.
- Forest polygons: remove any polygons that intersect with cocoa polygons and apply a 500m internal buffer to exclude the forest reserve edges.
- Merge cocoa and forest polygons into one layer, then rasterize with label values (0 = unknown, 1 = monocrop cocoa, 2 = forest, 3 = intercrop cocoa) using "mask_rasters" function.

### 2.2.3 Creating patches
"create_dataset" function:<br>
1. Initialize moving window at top left of study area (i.e. 128 x 128 pixels)
2. Move window across entire tile with a stride that is equal to the window size, in order to ensure no duplication of data between train and test datasets. For each location, check if labels satisfy the monocrop or intercrop dataset requirements.
3. If so, crop and save a copy of the label raster, save a corresponding cropped portion of the masked virtual raster.

## 2.3 Methods for machine learning
All relevant code in unet-code.ipynb

### 2.3.1 Architecture and hyperparameters
The U-NET architecture is adapted from the cocoa segmentation work of [Filella, 2018](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Student_Theses/BA_BonetFilella.pdf) and the Jupyter notebook U-NET implementation by [Bhatia,2021](https://github.com/VidushiBhatia/U-Net-Implementation) with the following parameters:<br>
- Input: 128 x 128 x number of bands
- Number of filters: 32
- Number of classes: 3
- Encoder: 5 blocks of two Conv Layers (3x3 filters, ’same’ padding) with relu activation and HeNormal initialization, max pooling
- Decoder: 4 blocks of transpose convolution, concatenate with skip connection from encoder, two Conv layers (3x3 filters, ’same’ padding)
- Model output: one Conv layer (3x3 filters, ’same’ padding) followed by one 1x1 convolution layer to get image to same size as input.

### 2.3.2 Data split
A 10% test split is applied using the sci-kit learn library (train test split function). The 10-fold validation and training data split is implemented using the sci-kit learn k-fold function, and a model is trained using 10 different splits. <br>
Note: for each training split, the weights leading to the best loss value are saved in a checkpoint file, therefore the model weights can be reloaded for inference at another time.

### 2.3.3 Testing
The trained model is used for inference of the test dataset, which results in a set of metrics being computed for each fold. The test images have been selected in regions of the study area with characteristics of interest and can be modified by adjusting the indices in the code. Additionally, several images from intercrop areas are also used to test the model. All metrics are output as a csv file. The inference is output as a probability map for cocoa (each pixel indicating the probability of belonging to the cocoa class) and a prediction mask (for each pixel, indicating the class with the highest probability). 

## 2.4 Methods for analysis
All relevant code in analysis.py

### 2.4.1 Plotting
Results from the output csv are copied to a new csv (e.g. "accuracy" columns for three different experiments) and can be visualized using the following functions:<br>
- **error_bars**: useful to visualize means, variation styled in a more aesthetic way
- **plot_boxplots**: useful to visualize data spread, median, standard deviation

### 2.4.1 Mapping
Both maps are output as rasters that can be visualized in QGIS and overlaid with polygons for visual analysis. 

## 2.5 Tools and libraries

### 2.5.1 Python IDE
Used Visual Studio Code. Installed the following libraries:
- Fiona
- geopandas
- numpy
- pandas
- rasterio
- shapely
- matplotlib

### 2.5.2 Jupyter notebook
Used AWS Studio Lab, Google Co-lab may also work. Installed the following libraries:
- numpy
- tensorflow
- keras
- matplotlib
- pandas
- gc
- tifffile
- scikit-learn
- gdal
- imageio
- opencv-python
- xeus-python
- rasterio

### 2.5.3 Other
- QGIS 3.28.6

# 3. Sharing / access information
## 3.1 Licenses/restrictions
This work is licensed under a Creative Commons Attribution 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/).

## 3.2 Recommended citation
Therias, A. (21 June 2023). Integrating radar and multi-spectral data to detect cocoa crops: a deep learning approach (Unpublished master's thesis). Delft University of Technology, The Netherlands. Retrieved from [http://resolver.tudelft.nl/uuid:314e12c9-c3bc-478b-b664-d0c0680f3caf](http://resolver.tudelft.nl/uuid:314e12c9-c3bc-478b-b664-d0c0680f3caf).

To cite the code, please refer to GitHub citation tool. 

# 4. Getting started
## 4.1 Local machine
1. Clone repository
2. Create virtual environment
3. pip install requirements.txt
4. Prepare ground truth polygon dataset in QGIS
5. Prepare satellite data raster(s) in QGIS
6. Create label raster: **mask_rasters** in preprocessing.py
7. Update parameters, dataset name and file paths. Create necessary folders following existing example.
8. Create training dataset: **create_dataset** preprocessing.py

## 4.2 Cloud computing
1. Setup account on AWS Studio Lab (or Google Colab)
2. Create "repository" and "thesis_data" folders.
3. In "repository," upload unet-code.ipynb and create new folder with dataset name (e.g. dec_MSI_jan_sar).
4. In "thesis_data," upload training data folder with the same name (e.g. dec_MSI_jan_sar) and including subfolders: mask, satellite, inter_mask, inter_satellite containing training patches.
5. Uncomment installation code at the top of unet-code.ipynb for the first time running.
6. Before each experiment, update parameters
- dataset
- n_classes
- n_channels
- size