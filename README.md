# A-RoF-Linearization-Dataset-Repository

This repository hosts a comprehensive dataset tailored for research in analog radio over fiber (A-RoF) systems, focusing particularly on linearization techniques. In A-RoF systems, mitigating nonlinear distortions is crucial, and while machine learning (ML)-based linearization schemes have been explored extensively, the lack of publicly available datasets hampers research progress. 

Here, we address this gap by offering a rich dataset. Leveraging a software-defined radio (SDR) approach, we generate digital samples of the generalized frequency division multiplexing (GFDM) waveform, which are then applied to a real A-RoF system. 

The dataset provided includes both transmitted and received GFDM waveform samples, enabling researchers effectively training ML models. By facilitating access to this dataset, we aim to foster the development of robust linearization schemes for A-RoF systems. Researchers can utilize this resource to enhance the effectiveness and robustness of ML-based linearization techniques, thus advancing the state-of-the-art in linearization A-RoF technology. 

*********************

## Summary :clipboard:
* [Dataset Information](#dataset-information)
* [How to Use](#how-to-use)
* [Example Usage](#example-usage)


*********************

 ## Dataset Information 📊 <a name="dataset-information"></a>

 ### Dataset Description:

The dataset provided here contains training instances and labels acquired from a practical A-RoF (Analog Radio-over-Fiber) system. Signals with bandwidths of 3, 6, 12, and 24 MHz were considered during data collection. For each bandwidth, RF power was varied from 0 to 11 dBm, resulting in eleven pairs of datasets for each bandwidth.

### File Format:

The files are saved in binary format, compatible with most Python IDEs.

### File Name Convention:

Each pair of datasets follows the naming convention:

   *   **Transmitted Samples:** (`dataset_n_rof_input_xdBm_yMHz`)
   *   **Received Samples:** (`dataset_n_rof_output_xdBm_yMHz`)

Where:

   * `n` varies from 1 to 11
   * `x` varies from 0 to 10
   * `y` represents the bandwidth, which can take values of 3, 6, 12 or 24 MHz

*********************

##  How to Use :arrow_forward: <a name="how-to-use"></a>

To utilize a dataset, follow these steps:

   * **Selecting Data Files:** Choose the transmitted file (`dataset_n_rof_input_xdBm_yMHz`) and its corresponding received file (`dataset_n_rof_output_xdBm_yMHz`).
  

   * **Loading Data:** Load the selected files into your Python environment using appropriate functions to read binary data.

   * **Processing:** Once loaded, process the data as required for your A-RoF linearization task.

### Example of How Load a Dataset in a Python IDE

After downloading the dataset from this repository, be sure to substitute 'path_to_tx_data' and 'path_to_rx_data' with the corresponding file paths on your system. Below, you'll find an example demonstrating how to load the transmitted dataset. This example specifically refers to the sixth dataset within the 6 MHz signa with 5 dBm RF power:

```python

import numpy as np

# Replace 'path_to_tx_data' and 'path_to_rx_data' with the actual file paths on your system
tx_data_path = 'path_to_tx_data/dataset_6_rof_input_5dBm_6MHz'
rx_data_path = 'path_to_rx_data/dataset_6_rof_output_5dBm_6MHz'

# Load the transmission data from the file
tx_data = np.fromfile(tx_data_path, dtype=np.complex64)

# Load the received data from the file
rx_data = np.fromfile(rx_data_path, dtype=np.complex64)


```
*********************

##  Example Usage <a name="example-usage"></a>

Below is a straightforward code snippet demonstrating how to load the dataset and train a basic neural network for linearizing an A-RoF system. In this example, a dataset is loaded, and a Multi-Layer Perceptron Neural Network (MLP) is utilized to estimate the inverse response of the A-RoF system. Here, the received samples serve as the neural network inputs, while the transmitted samples act as the neural network labels. This allows for learning the post-inverse response of the A-RoF system. Additionally, both the neural network weights and structure are saved, facilitating the reconstruction of the neural network inside a Software-Defined Radio (SDR) transceiver. This enables the implementation of a pre-distortion processing block.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:47:17 2024

@author: Luiz Augusto Melo Pereira
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten 
import matplotlib.pyplot as plt

# Earling Stop Technique
callback = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',  patience=100,min_delta=1e-9, verbose=1,restore_best_weights=True)

# Replace 'path_to_tx_data' and 'path_to_rx_data' with the actual file paths on your system
tx_data_path = '/home/luizmelo/Documentos/Dataset/6 MHz/dataset_6_rof_input_5dBm_6MHz'
rx_data_path = '/home/luizmelo/Documentos/Dataset/6 MHz/dataset_6_rof_output_5dBm_6MHz'

# Load the transmission data from the file
tx_data = np.fromfile(tx_data_path, dtype=np.complex64)

# Load the received data from the file
rx_data = np.fromfile(rx_data_path, dtype=np.complex64)

#---------------------------------
# Preparing the training data-set 
#--------------------------------
train_matrix = np.c_[rx_data.real, rx_data.imag]
train_labels = np.c_[tx_data.real, tx_data.imag]

#-----------------------
# Designing the MLP Neural Network
#----------------------
initializer = tf.keras.initializers.glorot_normal(seed=25)
mlpModel = Sequential()
mlpModel.add(Flatten(input_shape=(train_matrix.shape[1],)))
mlpModel.add(Dense(32, activation='relu',kernel_initializer=initializer))
mlpModel.add(Dense(16, activation='relu',kernel_initializer=initializer))
mlpModel.add(Dense(2,kernel_initializer=initializer))

#----------------------
# Compile the model
#----------------------
mlpModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
mlpModel.summary()
history = mlpModel.fit(train_matrix, train_labels,validation_split=0.3, epochs=5000, batch_size=1024, callbacks=[callback], verbose=2, shuffle=True)

#----------------------
# Saving both the structure and weights of the trained MLP neural network
#----------------------
mlpModel.save("mlpModel.h5")

mlpModel_json = mlpModel.to_json()
with open("mlpModel.json", "w") as json_file:
    json_file.write(mlpModel_json)


# summarize history for loss
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model loss')
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

```
