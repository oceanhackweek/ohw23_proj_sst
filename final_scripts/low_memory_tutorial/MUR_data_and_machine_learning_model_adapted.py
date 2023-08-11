'''
Developed by Juirai, Boris, Hao, combined by Paula Birocchi 

date: 09-08-2023

Ocean Hack Week 2023

This script was developed to run a machine learning model to predict SST surface distribution.

'''
# getting the libraries necessary to run this script:
import pandas as pd
from pathlib import Path
import xarray as xr
import numpy as np
import calendar
import os.path
import dask.array as da
from dask.delayed import delayed
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers, optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, Dropout, Dense, Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# Core libraries for this tutorial
from eosdis_store import EosdisStore # Available via `pip install zarr zarr-eosdis-store`
import requests
from pqdm.threads import pqdm
from matplotlib import animation, pyplot as plt
from IPython.core.display import display, HTML
from pprint import pprint
# importing library for s3 buckets
import s3fs


import dask.array as da
from dask.delayed import delayed
from sklearn.model_selection import train_test_split
import gc
import numpy as np
import matplotlib.pyplot as plt
    


def preprocess_day_data(day_data):
    day_data = da.squeeze(day_data)
    mean_val = da.nanmean(day_data).compute()  # compute here to get scalar value
    return day_data - mean_val

def preprocess_data(zarr_ds, chunk_size=200):
    total_len = zarr_ds['analysed_sst'].shape[0]
    chunk_shape = (chunk_size,) + zarr_ds['analysed_sst'].shape[1:]  # Adjusted chunking
    chunks = []

    for start_idx in range(0, total_len, chunk_size):
        end_idx = min(start_idx + chunk_size, total_len)
        
        # Directly slice the dask array without wrapping it with da.from_array again
        chunk = zarr_ds['analysed_sst'][start_idx:end_idx]
        
        processed_chunk = chunk.map_blocks(preprocess_day_data)
        
        # Use da.where to replace NaNs with 0.0
        processed_chunk = da.where(da.isnan(processed_chunk), 0.0, processed_chunk)
        
        chunks.append(processed_chunk)

    return da.concatenate(chunks, axis=0)
    #processed_data = preprocess_data(dscut)

def prepare_data_from_processed(processed_data, window_size=5): 
    length = processed_data.shape[0]
    X, y = [], []

    for i in range(length - window_size):
        X.append(processed_data[i:i+window_size])
        y.append(processed_data[i+window_size])

    X, y = da.array(X), da.array(y)
    return X, y
    #X, y = prepare_data_from_processed(processed_data)


def time_series_split(X, y, train_ratio=0.7, val_ratio=0.2):
    total_length = X.shape[0]
    
    # Compute end indices for each split
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    #X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)



# please change the number of points in your matrix before running the following code:
def create_simple_model(input_shape=(5, 201,201, 1)):
    model = Sequential()

    # ConvLSTM layer
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         input_shape=input_shape,
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    
    # Conv2D layer for output
    model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear'))

    return model




def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    # Self attention
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs, inputs)
    attn_output = tf.keras.layers.Add()([attention, inputs])
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_output)
    
    # Feed-forward network
    ffn_output = tf.keras.models.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(d_model),
    ])(out1)
    out2 = tf.keras.layers.Add()([ffn_output, out1])
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2)

def create_transformer_model(input_shape=(5, 149, 181, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    dim_out = [int((input_shape[1] + 1)/2), int((input_shape[2] + 1)/2)]
    
    # ConvLSTM layer with fewer filters
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3),
                                   padding='same', return_sequences=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Asymmetric padding after ConvLSTM
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    
    # Max pooling to reduce spatial dimensions
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Transformer layer with fewer dimensions
    d_model = 16
    num_heads = 2
    ff_dim = 32
    x = tf.keras.layers.Reshape((-1, d_model))(x)
    x = transformer_encoder(x, d_model, num_heads, ff_dim)
    
    x = tf.keras.layers.Reshape((dim_out[0], dim_out[1], d_model))(x)
    #x = tf.keras.layers.Reshape((75, 91, d_model))(x)
    
    # Upsample layer to match desired output size
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    # Cropping layer to match the exact desired size
    x = tf.keras.layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
    
    # Output Conv2D layer
    outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)








def create_land_mask(data): 
    land_mask = np.isnan(data)
    return np.flipud(land_mask)



def preprocess_vis_input_data(day_data):
    day_data = np.squeeze(day_data)
    mean_val = np.nanmean(day_data)
    processed_data = day_data - mean_val
    # Replace NaNs with 0.0
    processed_data = np.where(np.isnan(processed_data), 0.0, processed_data)
    return processed_data


def postprocess_prediction(prediction, input_data,land_mask_resized):
    # Find positions where the last day of input_data is 0
    
    # Set those positions in the prediction to NaN
    prediction[land_mask_resized] = np.nan
    
    # Add back the historical mean
    mean_val = np.nanmean(input_data)
    prediction = np.where(np.isnan(prediction), np.nan, prediction + mean_val)
    
    return prediction


def predict_and_plot(date_to_predict, window_size, model, dataset, land_mask_resized, plot=True):
    # Step 1: Select the time window
    time_index = np.where(dataset['time'].values == np.datetime64(date_to_predict))[0][0]
    input_data_raw = dataset['analysed_sst'][time_index-window_size:time_index].values
    true_output_raw = dataset['analysed_sst'][time_index].values
    print(input_data_raw.shape)
    print(true_output_raw.shape)
    # Preprocess the input data
    input_data = np.array([preprocess_vis_input_data(day) for day in input_data_raw])
    
    # Step 2: Make prediction
    prediction = model.predict(input_data[np.newaxis, ...])[0]
    
    # Postprocess the prediction
    prediction_postprocessed = postprocess_prediction(prediction, input_data_raw, land_mask_resized)
    print(prediction_postprocessed.shape)
    # Step 3: Visualize
    if plot:
        # Determine common scale for all plots
        input_data_raw = input_data_raw[..., np.newaxis]
        true_output_raw = true_output_raw[np.newaxis, ..., np.newaxis]
        prediction_postprocessed = prediction_postprocessed[np.newaxis, ...]
        
        all_data = np.concatenate([input_data_raw, prediction_postprocessed, true_output_raw])
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
        
        def plot_sample(sample,i, title=''):
            sample_2d = np.squeeze(sample)
            plt.imshow(sample_2d, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title(title)
            plt.colorbar()
            plt.savefig('figure1_test'+str(i)+'.png', bbox_inches='tight')
            plt.close()

        # show input frames
        for i, frame in enumerate(input_data_raw):
            plot_sample(frame, i,title=f'Input Frame {i+1} ({dataset["time"].values[time_index-window_size+i]})')
        
        # show predicted output
        plot_sample(prediction_postprocessed,i+1, title=f'Predicted Output ({date_to_predict})')
        
        # show true output
        plot_sample(true_output_raw, i+2,title=f'True Output ({date_to_predict})')

    return input_data_raw, prediction_postprocessed, true_output_raw
                


def compute_mae(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))



def load_the_dataset():
    # Bypass AWS tokens, keys etc.
    s3 = s3fs.S3FileSystem(anon=True)

    # Verify that we're in the right place
    # getting the sattelite data from S3 buckets: 
    sst_files = s3.ls("mur-sst/zarr-v1/")
    sst_files

    ds = xr.open_zarr(
            store=s3fs.S3Map(
                root=f"s3://{sst_files[0]}", s3=s3, check=False
            )
    )
    return ds


def pull_a_tile(ds, lat, lon, time_slice):
    # We reduced the size of matrix to test the model, as we were having issues:
    # slicing our data to be able to make the model run:
    dscut = ds.sel(time=time_slice,lat=slice(lat[0],lat[1]),lon=slice(lon[0],lon[1]))
    print('lat {}'.format(lat))
    print('lon {}'.format(lon))
    #dscut = ds.sel(time=slice("2002-06-01", "2002-06-30"),lat=slice(5,7),lon=slice(50,52))

    dscut['time'] = dscut['time'].dt.floor('D')

    # processed_data = preprocess_data(zarr_ds).compute()
    processed_data = preprocess_data(dscut)

    X, y = prepare_data_from_processed(processed_data)

    #Split the X, y into train/val/test
    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)
    return dscut, X_train, y_train, X_val, y_val, X_test, y_test


def explore_results(X, model, dscut):
    #PART 2:
    #Look at the results:
    
    #land_mask_resized = create_land_mask(X[0][0].compute())    
    land_mask_resized = create_land_mask(X[0][0].compute()) #Prepare a land mask
    np.save('land_mask_resized.npy', land_mask_resized)
    
    #Implement a Prediction
    date_to_predict = '2002-06-30'
    window_size = 5
    input_data, predicted_output, true_output = predict_and_plot(date_to_predict, window_size, model, dscut, land_mask_resized)
    
    predicted_mae = compute_mae(true_output, predicted_output)
    print(f"MAE between Predicted Output and True Output: {predicted_mae}")
    
    last_input_frame = input_data[-1]
    last_input_frame_2d = np.squeeze(last_input_frame)
    true_output_2d = np.squeeze(true_output)
    last_frame_mae = compute_mae(true_output_2d, last_input_frame_2d)
    print(f"MAE between Last Input Frame and True Output: {last_frame_mae}")
    
    #model.save('ConvLSTM_nc_2002-.keras')
    
    # just plotting land mask
    data = np.load('land_mask_resized.npy')
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.title('Land Mask')
    plt.savefig('land_mask_version2.png')
    #plt.close()
    
    
def run():
    
    #Load the SST data
    ds = load_the_dataset()

    #Time slice to use for training:
    time_slice = slice("2002-06-01", "2002-06-30")

    #Overall lat/lon area for the study
    lat_all = [-4, 32]
    lon_all = [44, 90]

    #Set the model hyper parameters
    batch_size = 64 #Data batch to load in
    step_size = 2 #size of the square, degrees lat by degrees lon

    num_tiles = 10 #This is the number of times it selects different input lat/lon tiles
    num_epochs = 1 # This is the number of training epochs per tile, so don't want this to be too large

    lats = np.arange(lat_all[0], lat_all[1], step_size)  
    lons = np.arange(lon_all[0], lon_all[1], step_size)

    #Set the outputs
    parent_folder = os.getcwd()
    model_path = parent_folder + '/sst_model'
    
    # Rather than stepping through the lat and lon in 2 loops, randomly select the lat and lon tiles
    run_number = 0 # initialize this so the model knows to initialize
    for num in np.arange(0,num_tiles):

        #Randomly select lat/lon tiles
        lat = random.randint(lat_all[0],lat_all[1]-step_size)
        lon = random.randint(lon_all[0],lon_all[1]-step_size)
        lat = [lat, lat+step_size]
        lon = [lon, lon+step_size]

        #Load the tile:
        print(f'{lat}, {lon}')
        X_train, y_train, X_val, y_val, X_test, y_test = pull_a_tile(ds,lat,lon,time_slice)[1:]

        if run_number == 0:
            #Compile the model:

            #model = create_transformer_model(np.shape(X_train)[1:] + (1,))

            model = create_simple_model(np.shape(X_train)[1:] + (1,))

            #model = create_simple_model()
            model.summary()
            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            early_stop = EarlyStopping(patience=5, restore_best_weights=True)


        #Train the model:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(32)

        history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, callbacks=[early_stop])

        #Increment the run number
        run_number += 1

        #Delete the data to free up memory:
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()

    #Save the model
    model.save(model_path)

