# GeoSci Seismic Data Classification

## Introduction
This Git repo stores the source code to build the neural network for image classification of geographical science images of mining seismic data. The neural network build here, is a Convolutional Neural Network (CNN).

## Requirements

Ensure that the following libraries are installed before 
- tensorflow
- PIL
- numpy
- matplotlib

Ideally, you should setup a virtual environment and have this repo cloned in your virtual environment. 

```bash
git clone https://github.com/aliceyang16/GeoSci_Seismic_Data_classification.git
```

**Note**: When installing tensorflow on a windows machine, its best to install it though anaconda. 

Using anaconda, you will need to activate the tf virtual environment that you have installed the tensorflow library in using the following:

```bash
conda create -n venv tensorflow-gpu
conda activate venv
```