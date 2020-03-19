#Import A TensorFlow Model And Run Inference

## Running the MNIST sample

1.  Compile MNIST sample by running `make` in the `<TensorRT root directory>/samples/sampleUffMNIST` directory. The binary named `sample_uff_mnist` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleUffMNIST
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to create an MNIST engine from a UFF model and perform inference using it.
	```
	./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8] [--fp16]



## Running the MLP sample

