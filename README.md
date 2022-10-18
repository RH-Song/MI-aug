# MI Aug

## Description
ECG Augmentation Method for Automatic Detection of ST-Segment Elevation Myocardial Infarction and Culprit Vessel

## Requirement
1.  PyTorch == 1.8.2 

## Run
1. Train the network with
```
python NewTestFor2Dir.py
```
2. If modifying model-cfg for your own dataset, you might need to modify the num_classes of the neural networks and the input channels of the first layer.

## Log and plot
```
python NewTestFor2Dir.py > log.txt 2>&1
python plot_log.py log.txt 
```