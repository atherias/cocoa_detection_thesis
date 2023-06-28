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
- [3. FILE OVERVIEW](#3-file-overview)
  - [3.1 File List](#31-file-list)
- [5. SHARING/ACCESS INFORMATION](#5-sharingaccess-information)
  - [5.1 Licenses/restrictions placed on the data](#51-licensesrestrictions-placed-on-the-data)
  - [5.2 Links to other resources](#52-links-to-other-resources)
  - [5.3 Recommended citation for this dataset](#57-recommended-citation-for-this-dataset)


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

Research question: To what extent can a CNN trained with multispectral and SAR datasets enable the automated detection of cocoa crops in Ghana?
Sub-questions
- How does the combination of MSI and SAR data affect the results of cocoa parcel segmentation trained with data from a single day?
- How does the combination of MSI and SAR data affect the results of cocoa parcel segmentation trained with temporal datasets?
- Why does the use of different polarizations (i.e. Vertical-Vertical (VV) or Vertical-Horizontal (VH)) affect the influence of SAR datasets on the cocoa segmentation results?
- What is the impact of SAR and MSI training data on the detection of intercrop cocoa?

## 2.2 Methods for processing the data
All relevant functions in preprocessing.py

### 2.2.1 Satellite data
S1 and S2 datasets are downloaded via the WEkEO JupyterHub (Earth Observation Server) using the WEkEO Harmonized Data Access API. The MSI datasets (Level 2A) are filtered to contain less than 15% cloud cover and efforts are made to select imagery distributed across the wet and dry seasons. The year 2020-2021 is selected because it contains datasets that match as closely as possible to an even distribution across the seasons, and because this timeline overlaps with the ground truth dataset collection dates. The SAR datasets (Level 1C) are filtered to IW mode and GRD products only. For the dry season stack, the images are selected as close as possible to the dates of MSI data.
- Project MSI data to UTM zone 30N and re-sample all 20 m resolution bands to 10 m resolution using bi-linear interpolation
- Project SAR to UTM zone 30N, re-sample to 10 m resolution and clip to study area
- Stack MSI and SAR in various virtual rasters with each file saved as a separate band.

### 2.2.2 Polygon data
After collecting the ground truth polygons from Meridia, the following steps are conducted using QGIS:
- Quality-control ground truth polygons: conduct visual checks on cocoa ground truth to check for possible misclassification, such as overlaps between cocoa and non-cocoa ground truth
- Cocoa polygons: remove any polygons classified as ”sparse,” ”unknown” or ”both” to keep only ”intercrop” and ”monocrop” cultivation types. Remove any intercrop polygons intersecting with monocrop polygons and vice-versa.
- Forest polygons: remove any polygons that intersect with cocoa polygons and apply a 500m internal buffer to exclude the forest reserve edges.
- Merge cocoa and forest polygons into one layer, then rasterize with label values (0 = unknown, 1 = monocrop cocoa, 2 = forest, 3 = intercrop cocoa) using "mask_rasters" function.

### 2.2.3 Creating patches
"create_dataset" function:
1. Initialize moving window at top left of study area (i.e. 128 x 128 pixels)
2. Move window across entire tile with a stride that is equal to the window size, in order to ensure no duplication of data between train and test datasets. For each location, check if labels satisfy the monocrop or intercrop dataset requirements.
3. If so, crop and save a copy of the label raster, save a corresponding cropped portion of the masked virtual raster.

## 2.3 Methods for machine learning
All relevant code in unet-code.ipynb

### 2.3.1 Architecture and hyperparameters
The U-NET architecture is adapted from the cocoa segmentation work of [Filella, 2018](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Student_Theses/BA_BonetFilella.pdf) and the Jupyter notebook U-NET implementation by [Bhatia,2021](https://github.com/VidushiBhatia/U-Net-Implementation) with the following parameters:
- Input: 128 x 128 x number of bands
- Number of filters: 32
- Number of classes: 3
- Encoder: 5 blocks of two Conv Layers (3x3 filters, ’same’ padding) with relu activation and HeNormal initialization, max pooling
- Decoder: 4 blocks of transpose convolution, concatenate with skip connection from encoder, two Conv layers (3x3 filters, ’same’ padding)
- Model output: one Conv layer (3x3 filters, ’same’ padding) followed by one 1x1 convolution layer to get image to same size as input.

### 2.3.2 Data split
A 10% test split is applied using the sci-kit learn library (train test split function). The 10-fold validation and training data split is implemented using the sci-kit learn k-fold function, and a model is trained using 10 different splits.

## 2.4 Methods for analysis
All relevant code in analysis.py


