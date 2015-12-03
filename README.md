# 91-523-f15-project
Project for 91.523 Computer Vision, Fall 2015

To use the following code, you will need a working build of Caffe.

This project consists of 3 important pieces of code. 
1. classification.cpp 
  This file is intended to be used instead of the existing classification.cpp bundled with Caffe, located under examples/cpp_classification/classification.cpp. Replace the bundled file with this one, and rebuild the Caffe project. The resulting executable created can then be used with the models trained using the following perl scripts.

2. createFullTuningCNNFiles.pl
  This perl script creates the files necessary for, and then trains, a fully fine-tuned convolution neural network for the given command line parameters
  
3. createLastTwoLayerTuningCNNFiles.pl
  This perl script creates the files necessary for, and then trains, a fine-tuned convolution neural network, with only the last two layers fine-tuned, for the given command line parameters
