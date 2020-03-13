# TensorRtOptimization
Special Problem Georgia Tech. Optimize Deep Learning Model by unsing TensorRT and cuda.


## Install required libraries
All needed libraries are in the 'requirements.txt' file.
To install run the following command on a terminal:
-  in python3:`python3 -m pip install -r requirements.txt`
-  in python2:`python2 -m pip install -r requirements.txt`


##Jetson Xavier AGX

### Working Environment: Jetpack 4.2.x
-  Ubuntu  18.04.3 LTS : lsb_release -a
-  Python 3.6.9 : python3 --version
-  TensorFlow 1.15.0 : python3 -c 'import tensorflow as tf; print(tf.__version__)'
-  TensorRT 7.0.6.3
-  Cuda 10.0 : nvcc --version (gives cuda compiler version --> to build h5py)


### Install a particular version of tensorflow-gpu for a specific jetson version
General Command : pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow-gpu==$TF_VERSION+nv$NV_VERSION

Replace the following on the we link :
-   JP_VERSION : jetson packet version (42 for JetPack 4.2.x and  33 for Jetpack 3.3.x)
-   TF_VERSION : tensorflow-gpu version (for example 1.15.0)
-   NV_VERSION : NVIDIA container version (for example 19.01)

Command executed for JetPack 4.2.x to get tensorFlow-gpu==1.15.0:
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.15.0 


### Install required libraries
All needed libraries are in the 'requirements.txt' file:
-  numpy
-  Pillow
-  pycuda
-  common
-  cython (for tensorflow user)


To install run the following command on a terminal:
-  in python3:`python3 -m pip install -r requirements.txt`
-  in python2:`python2 -m pip install -r requirements.txt`


## Launch python code
python inferJetsonXavier.py
