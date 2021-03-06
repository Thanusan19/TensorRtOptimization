/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleUffMNIST.cpp
//! This file contains the implementation of the Uff MNIST sample.
//! It creates the network using the MNIST model converted to uff.
//!
//! It can be run with the following command line:
//! Command: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>


const std::string gSampleName = "TensorRT.sample_uff_mnist";

//!
//! \brief  The SampleUffMNIST class implements the UffMNIST sample
//!
//! \details It creates the network using a Uff model
//!
class SampleUffMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleUffMNIST(const samplesCommon::UffSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! Calculate the aearage time on 100 * 1000 inferences (MLP model engine)
    //!
    bool infer2();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief Parses a Uff model for MNIST and creates a TensorRT network
    //!
    void constructNetwork(
        SampleUniquePtr<nvuffparser::IUffParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result
    //!        in a managed buffer
    //!
    bool processInput(
        const samplesCommon::BufferManager& buffers, const std::string& inputTensorName) const;

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(
        const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const;


    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    samplesCommon::UffSampleParams mParams;

    nvinfer1::Dims mInputDims;
    const int kDIGITS{10};
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by parsing the Uff model
//!          and builds the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleUffMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }
    constructNetwork(parser, network);
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1000_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a Uff parser to create the MNIST Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
void SampleUffMNIST::constructNetwork(
    SampleUniquePtr<nvuffparser::IUffParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // There should only be one input and one output tensor
    assert(mParams.inputTensorNames.size() == 1);
    assert(mParams.outputTensorNames.size() == 1);

    // Register tensorflow input
    parser->registerInput(
        mParams.inputTensorNames[0].c_str(), nvinfer1::Dims3(1, 12, 5), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    parser->parse(mParams.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    if (mParams.int8)
    {
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer
//!
bool SampleUffMNIST::processInput(
    const samplesCommon::BufferManager& buffers, const std::string& inputTensorName) const
{
    const int inputH = 12; //mInputDims.d[1];
    const int inputW = 5;//mInputDims.d[2];

    //Import and convert input data from X.txt to a vector Xfloat//
    ifstream f("/home/localadmin/NNOptimization/TensorRT-7.0.0.11/samples/sampleUffMNIST/X.txt");
    string Xstr;
    std::vector<float> Xfloatv;
    while(getline(f,Xstr,',')){
	float Xfloat= std::stof(Xstr);
	Xfloatv.push_back(Xfloat);
    }
    f.close();

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    for (int i = 0; i < inputH * inputW * mParams.batchSize ; i++)
    {
        hostInputBuffer[i] =Xfloatv[i]; //fileData[i];
    }
    return true;
}




//!
//! \brief Verifies that the inference output is correct
//!
bool SampleUffMNIST::verifyOutput(
    const samplesCommon::BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    gLogInfo << "Output:\n";

    float val{0.0f};
    int idx{0};

    // Determine index with highest output value
    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }
    }

    // Print output values for each index
    for (int j = 0; j < kDIGITS; j++)
    {
        gLogInfo << j << "=> " << setw(10) << prob[j] << "\t : ";

        // Emphasize index with highest output value
        if (j == idx)
        {
            gLogInfo << "***";
        }
        gLogInfo << "\n";
    }

    gLogInfo << std::endl;
    return (idx == groundTruthDigit);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample.
//!  It allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleUffMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    bool outputCorrect = true;
    float total = 0;
    int inferenceLoop=1000;

    // Try to infer 
    for (int index = 0; index < inferenceLoop; index++)
    {
        if (!processInput(buffers, mParams.inputTensorNames[0]))
        {
            return false;
        }
        // Copy data from host input buffers to device input buffers
        buffers.copyInputToDevice();

        const auto t_start = std::chrono::high_resolution_clock::now();

        // Execute the inference work
        if (!context->execute(mParams.batchSize, buffers.getDeviceBindings().data()))
        {
            return false;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;

        // Copy data from device output buffers to host output buffers
        buffers.copyOutputToHost();

        // Check and print the output of the inference
        //outputCorrect &= verifyOutput(buffers, mParams.outputTensorNames[0], digit);
    }

    //total /= inferenceLoop;

    gLogInfo << "Average time is " << total << " ms." << std::endl;

    //return outputCorrect;
    return true;
}



//!
//! \brief Runs the TensorRT inference engine for MLP and calculate average time on 100 * 1000 inferences
//!
//! \details This function is the main execution function of the sample.
//!  It allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleUffMNIST::infer2()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    bool outputCorrect = true;
    float total = 0;
    int inferenceLoop1=1000;
    int inferenceLoop2=100;
    int timeList[inferenceLoop2];

    // Try to infer 
    for (int i = 0; i < inferenceLoop2; i++)
    {
	gLogInfo<< "Lot of 1000 inferences Index = "<<i<<std::endl;
	total=0;

        if (!processInput(buffers, mParams.inputTensorNames[0]))
        {
            	return false;
        }
        // Copy data from host input buffers to device input buffers
        buffers.copyInputToDevice();

	for (int j = 0; j < inferenceLoop1; j++){
                const auto t_start = std::chrono::high_resolution_clock::now();
        	// Execute the inference work
        	if (!context->execute(mParams.batchSize, buffers.getDeviceBindings().data()))
        	{
            		return false;
        	}

        	const auto t_end = std::chrono::high_resolution_clock::now();
        	const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        	total += ms;
	}

       	// Copy data from device output buffers to host output buffers
       	buffers.copyOutputToHost();
	timeList[i]=total;
    }

    //Calculate the average
    float sum=0.0;
    float aver=0.0;
    for(int i=0;i<inferenceLoop2;i++){
	sum+=timeList[i];
    }
    aver=sum/inferenceLoop2;

    //Calcuate statistical variance
    float V=0;
    for(int i=0;i<inferenceLoop2;i++){
	V = V +  (timeList[i] - aver)*(timeList[i] - aver);
    }
    V=V/inferenceLoop2;


    gLogInfo << "Average time is: " << aver << " ms." << std::endl;
    gLogInfo << "Variance is: " << V <<std::endl; 

    return true;
}




//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleUffMNIST::teardown()
{
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Initializes members of the params struct
//!        using the command line args
//!
samplesCommon::UffSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::UffSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.uffFileName = locateFile("model.uff", params.dataDirs);
    params.inputTensorNames.push_back("inputs");
    params.batchSize = 2000;
    params.outputTensorNames.push_back("output/BiasAdd");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_mnist [-h or --help] [-d or "
                 "--datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding "
                 "the default. This option can be used multiple times to add "
                 "multiple directories. If no data directories are given, the "
                 "default is to use (data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, where n is the number of "
                 "DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    samplesCommon::UffSampleParams params = initializeSampleParams(args);

    SampleUffMNIST sample(params);
    gLogInfo << "Building and running a GPU inference engine for Uff MNIST" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer2())
    {
        return gLogger.reportFail(sampleTest);
    }
    /*if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }*/

    return gLogger.reportPass(sampleTest);
}
