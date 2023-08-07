# OHW 2023 - Predicting SST using satellite SST data and machine learning

Project idea: SST spatial distribution prediction using machine learning

We have a SST time series from 2000 to 2020 from ERA5 reanalysis to start to work with the model, but we are also interested in obtaining and using sattelite data from MUR (https://urs.earthdata.nasa.gov/).

Pitch + Ideation: Predict SST anomalies (upwelling, other interesting SST anomalies), generate SST spatial distribution forecast. The machine learning model developed here can also be used with other type of data as input! 


# STEPS:


1) Data, boundary box (time [2000-2020], lat [-5,32], lon[45,90])
2) Split data on training, validation and testing datasets
3) Model Architecture: 
a) ConvLSTM: this is going to be our first approach!
b) 3D CNN
c) Transformers
d) Hybrid: CNN + Transformer + LSTM
e) * SHAPE CORRECT

4) Complie and fit
a) Early-stop
b) * Loss function: MSE, MAE, SSIM
c) * metric

5) Visualization of result and Interpretation! 


# Ideation board: https://jamboard.google.com/d/1lOgVwnqQLvNRPAOEVEGnWXm8FSTuPYQWbteptKrslTM/viewer?f=10
# Slack channel: ohw23_proj_sst

# Build the project team:
Github repo: URL: https://github.com/oceanhackweek/ohw23_proj_sst/

# Team members:
Jiarui Yu, Boris Shapkin, Paula Birocchi (Seattle, US), ? Does anyone want to join us?
# Jam Board:
https://jamboard.google.com/d/1lOgVwnqQLvNRPAOEVEGnWXm8FSTuPYQWbteptKrslTM/viewer?f=10

One of the goals of this project could be detecting upwelling areas:
![Screenshot 2023-06-26 at 5 52 43 PM](https://github.com/oceanhackweek/ohw23_proj_sst/assets/25447814/662fbb25-601c-4e2f-b733-da9d7051d7a6)



