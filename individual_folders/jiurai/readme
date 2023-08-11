## Model Structures for Predicting SST Distribution

This folder provides three model structures designed to predict SST distribution based on past `n` days (default `n = 5`):

- **Transformer**
- **ConvLSTMs**
- **Transformer + ConvLSTM**

### Upwelling Classification

Once the predicted SST image data is available, we developed a subsequent classification model to determine whether upwelling occurs within that SST data.

> :warning: **Notice**: The labels utilized in the classification model are derived by comparing SST values between near-shore and off-shore points within our designated bounding box. If you wish to adopt a different labeling technique, you can employ the approach highlighted in the file `'upwelling_classification_nc'` to create a classification model tailored to upwelling.

### Reproduction Notes

To accurately replicate our time-series models:

- Ensure your environment is equipped with ample RAM.
- Usage of a GPU is recommended.
- Be aware that training duration can vary based on several factors including model architecture, data volume, and data resolution.
