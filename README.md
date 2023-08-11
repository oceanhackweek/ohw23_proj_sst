# OHW 2023 

# Predicting SST spatial distribution using satellite SST data and deep learning models

# One-line Description
SST spatial distribution prediction using deep learning

# Collaborators and Roles

| Name                | Location   | Role                |
|---------------------|------------|---------------------|
| Joseph Gum          | Virtual    | Project Facilitator |
| Jiarui Yu           | Seattle    | Participant         |
| Paula Birocchi      | Seattle    | Participant         |
| Boris Shapkin       | Seattle    | Participant         |
| Hao Tang            | Australia  | Participant         |
| Danyang Li          | Australia  | Participant         |
| Chandrama Sarker    | Australia  | Participant         |
| Zhengxi Zhou        | Australia  | Participant         |
| Dafrosa Kataraihya  | Virtual    | Participant         |
| Alex Slonimer       | Virtual    | Participant         |

# Background

We have a SST time series from 2000 to 2020 from ERA5 reanalysis to start to work with the model, but we are also interested in obtaining and using sattelite data from MUR (https://urs.earthdata.nasa.gov/). The satellite data is available in the S3 bucket. You can easily access this dataset using this Python code:
[https://github.com/oceanhackweek/ohw23_proj_sst/blob/main/access_MUR_satellite_data_through_python_S3bucket.py](https://github.com/oceanhackweek/ohw23_proj_sst/blob/main/temp/access_MUR_satellite_data_through_python_S3bucket.py)

# Goals
Pitch + Ideation: Predict SST anomalies (upwelling, other interesting SST anomalies), generate SST spatial distribution forecast.
SST prediction is very important to understand the hydrodynamics and thermodynamics processes in the ocean and also near surface atmosphere-ocean interactions. 

# Datasets
MUR Satellite Data (2002-present): S3 bucket and NASA website: https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1

# Workflow/Roadmap

1) Get data, and define our bounding box/ area of interest (desired approach: time [2000-2020], lat [-5,32], lon[45,90]);
2) Split data on training, validation and testing datasets;
3) Model Architecture: 
a) ConvLSTM: this is going to be our first (and main) approach!
b) 3D CNN
c) Transformers
d) Hybrid: CNN + Transformer + LSTM ( we won't have enough time to apply this model)

4) Complie and fit
a) Early-stop
b) * Loss function: MSE, MAE, SSIM (until now, we found better results using MSE)
c) * metric

5) Visualization of result and Interpretation! 
# References


# Project idea
The deep learning model developed here can also be used with other type of data as input! The idea is to use this model with other parameters in the future. For now, we are interested in forecasting the next day of SST spatial distribution.

# Ideation board
https://jamboard.google.com/d/1lOgVwnqQLvNRPAOEVEGnWXm8FSTuPYQWbteptKrslTM/viewer?f=10
# Slack channel
ohw23_proj_sst

# Ideation Presentation
https://docs.google.com/presentation/d/1eQKSdFHNGMDqGJMY4d-yGnNm4UrUj5kIS2mLQGPMZC8/edit#slide=id.g239da66eda5_25_5

# Final presentation
https://docs.google.com/presentation/d/1uUAIsuj9bxOFMVeIG_h5Bs-ZGDrRodldlz2FHfj4TbE/edit#slide=id.p

# Project google drive
https://drive.google.com/drive/folders/1M0o_R4aoDxU9XJOtLEHe90bma-Jn-jM9
