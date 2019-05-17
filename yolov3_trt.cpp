#include <algorithm>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;


// origin params
samplesCommon::Args gArgs;

// Res params
string onnxFile = "./yolov3_xray.onnx";
string engineFile = "./yolov3_xray.trt";
string fileList = "./list.txt";

vector<string> labels = { "water", "KK", "knife" };
vector<vector<int> > output_shape = { {1, 24, 13, 13}, {1, 24, 26, 26}, {1, 24, 52, 52} };
vector<vector<int> > g_masks = { {6, 7, 8}, {3, 4, 5}, {0, 1, 2} };
vector<vector<float> > g_anchors = { {25, 50}, {50, 25}, {40, 80}, {117.3573, 108.6033}, {175.3990, 73.3345}, {90.4552, 222.0880}, {156.7773, 152.8589}, {240.6245, 104.3916}, {220.2752, 217.6105} };
float obj_threshold = 0.10;
float nms_threshold = 0.30;

int CATEGORY = 3;
int BATCH_SIZE = 1;
int INPUT_CHANNEL = 3;
int DETECT_WIDTH = 416;
int DETECT_HEIGHT = 416;

// Res struct & function
typedef struct DetectionRes {
	float x, y, w, h, prob;
} DetectionRes;

float sigmoid(float in) {
	return 1.f / (1.f + exp(-in));
}
float exponential(float in) {
	return exp(in);
}

float* merge(float* out1, float* out2, float* out3, int bsize_out1, int bsize_out2, int bsize_out3)
{
	float* out_total = new float[bsize_out1 + bsize_out2 + bsize_out3];

	for (int j = 0; j < bsize_out1; ++j)
	{
		int index = j;
		out_total[index] = out1[j];
	}

	for (int j = 0; j < bsize_out2; ++j)
	{
		int index = j + bsize_out1;
		out_total[index] = out2[j];
	}

	for (int j = 0; j < bsize_out3; ++j)
	{
		int index = j + bsize_out1 + bsize_out2;
		out_total[index] = out3[j];
	}
	return out_total;
}

vector<string> split(const string& str, char delim)
{
	stringstream ss(str);
	string token;
	vector<string> container;
	while (getline(ss, token, delim))
	{
		container.push_back(token);
	}

	return container;
}

list<string> readFileList(const string& fileName)
{
	list<string> fileList;

	ifstream file(fileName);
	if (!file.is_open())
	{
		cout << "read file error: " << fileName << endl;
	}

	string strLine;
	while (getline(file, strLine))
	{
		vector<string> line = split(strLine, '\n');
		if (line.size() < 1)
		{
			continue;
		}
		vector<string> strs = split(line[0], ' ');

		int idx = 0;
		string dataName = strs[idx++];
		fileList.push_back(dataName);
	}

	file.close();

	return fileList;
}

void DoNms(vector<DetectionRes>& detections, float nmsThresh) {
	auto iouCompute = [](float * lbox, float* rbox) {
		float interBox[] = {
			max(lbox[0], rbox[0]), //left
			min(lbox[0] + lbox[2], rbox[0] + rbox[2]), //right
			max(lbox[1], rbox[1]), //top
			min(lbox[1] + lbox[3], rbox[1] + rbox[3]), //bottom
		};

		if (interBox[2] >= interBox[3] || interBox[0] >= interBox[1])
			return 0.0f;

		float interBoxS = (interBox[1] - interBox[0] + 1) * (interBox[3] - interBox[2] + 1);
		return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
	};

	sort(detections.begin(), detections.end(), [=](const DetectionRes & left, const DetectionRes & right) {
		return left.prob > right.prob;
	});

	vector<DetectionRes> result;
	for (unsigned int m = 0; m < detections.size(); ++m) {
		result.push_back(detections[m]);
		for (unsigned int n = m + 1; n < detections.size(); ++n) {
			if (iouCompute((float *)(&detections[m]), (float *)(&detections[n])) > nmsThresh) {
				detections.erase(detections.begin() + n);
				--n;
			}
		}
	}
	detections = move(result);
}
vector<DetectionRes> postProcess(cv::Mat& image, float * output) {
	vector<DetectionRes> detections;
	int total_size = 0;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];
		int size = 1;
		for (int j = 0; j < shape.size(); j++) {
			size *= shape[j];
		}
		total_size += size;
	}

	int offset = 0;
	float * transposed_output = new float[total_size];
	float * transposed_output_t = transposed_output;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];  // nchw
		int chw = shape[1] * shape[2] * shape[3];
		int hw = shape[2] * shape[3];
		for (int n = 0; n < shape[0]; n++) {
			int offset_n = offset + n * chw;
			for (int h = 0; h < shape[2]; h++) {
				for (int w = 0; w < shape[3]; w++) {
					int h_w = h * shape[3] + w;
					for (int c = 0; c < shape[1]; c++) {
						int offset_c = offset_n + hw * c + h_w;
						*transposed_output_t++ = output[offset_c];
					}
				}
			}
		}
		offset += shape[0] * chw;
	}
	vector<vector<int> > shapes;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];
		vector<int> tmp = { shape[2], shape[3], 3, 8 };
		shapes.push_back(tmp);
	}

	offset = 0;
	for (int i = 0; i < output_shape.size(); i++) {
		auto masks = g_masks[i];
		vector<vector<float> > anchors;
		for (auto mask : masks)
			anchors.push_back(g_anchors[mask]);
		auto shape = shapes[i];
		for (int h = 0; h < shape[0]; h++) {
			int offset_h = offset + h * shape[1] * shape[2] * shape[3];
			for (int w = 0; w < shape[1]; w++) {
				int offset_w = offset_h + w * shape[2] * shape[3];
				for (int c = 0; c < shape[2]; c++) {
					int offset_c = offset_w + c * shape[3];
					float * ptr = transposed_output + offset_c;
					ptr[4] = sigmoid(ptr[4]);
					ptr[5] = sigmoid(ptr[5]);
					ptr[6] = sigmoid(ptr[6]);
					ptr[7] = sigmoid(ptr[7]);
					float score = max(ptr[4] * ptr[5], ptr[4] * ptr[6]);
					score = max(score, ptr[4] * ptr[7]);
					if (score < obj_threshold)
						continue;
					ptr[0] = sigmoid(ptr[0]);
					ptr[1] = sigmoid(ptr[1]);
					ptr[2] = exponential(ptr[2]) * anchors[c][0];
					ptr[3] = exponential(ptr[3]) * anchors[c][1];

					ptr[0] += w;
					ptr[1] += h;
					ptr[0] /= shape[0];
					ptr[1] /= shape[1];
					ptr[2] /= DETECT_WIDTH;
					ptr[3] /= DETECT_WIDTH;
					ptr[0] -= ptr[2] / 2;
					ptr[1] -= ptr[3] / 2;

					DetectionRes det;;
					det.x = ptr[0];
					det.y = ptr[1];
					det.w = ptr[2];
					det.h = ptr[3];
					det.prob = score;
					detections.push_back(det);
				}
			}
		}
		offset += shape[0] * shape[1] * shape[2] * shape[3];
	}
	delete[]transposed_output;

	int h = DETECT_WIDTH;   //net h
	int w = DETECT_WIDTH;   //net w

	//scale bbox to img
	int width = image.cols;
	int height = image.rows;
	float scale = min(float(w) / width, float(h) / height);
	float scaleSize[] = { width * scale, height * scale };

	//correct box
	for (auto& bbox : detections) {
		bbox.x = (bbox.x * w - (w - scaleSize[0]) / 2.f) / scale;
		bbox.y = (bbox.y * h - (h - scaleSize[1]) / 2.f) / scale;
		bbox.w *= w;
		bbox.h *= h;
		bbox.w /= scale;
		bbox.h /= scale;
	}

	//nms
	float nmsThresh = nms_threshold;
	if (nmsThresh > 0)
		DoNms(detections, nmsThresh);

	return detections;
}


// prepare img
vector<float> prepareImage(cv::Mat& img) {
	int c = 3;
	int h = DETECT_WIDTH;   //net h
	int w = DETECT_WIDTH;   //net w

	float scale = min(float(w) / img.cols, float(h) / img.rows);
	auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

	cv::Mat rgb;
	cv::cvtColor(img, rgb, CV_BGR2RGB);
	cv::Mat resized;
	cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

	cv::Mat cropped(h, w, CV_8UC3, 127);
	Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
	resized.copyTo(cropped(rect));

	cv::Mat img_float;
	cropped.convertTo(img_float, CV_32FC3, 1.f / 255.0);

	//HWC TO CHW
	vector<Mat> input_channels(c);
	cv::split(img_float, input_channels);

	vector<float> result(h * w * c);
	auto data = result.data();
	int channelLength = h * w;
	for (int i = 0; i < c; ++i) {
		memcpy(data, input_channels[i].data, channelLength * sizeof(float));
		data += channelLength;
	}
	return result;
}


// load engine file
bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream)  //output buffer for the TensorRT model
{
	using namespace std;
	fstream file;
	cout << "loading filename from:" << engineFile << endl;
	nvinfer1::IRuntime* trtRuntime;
	nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
	file.open(engineFile, ios::binary | ios::in);
	file.seekg(0, ios::end);
	int length = file.tellg();
	//cout << "length:" << length << endl;
	file.seekg(0, ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	cout << "load engine done" << endl;
	std::cout << "deserializing" << endl;
	trtRuntime = createInferRuntime(gLogger.getTRTLogger());
	ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
	cout << "deserialize done" << endl;
	trtModelStream = engine->serialize();

	return true;
}



void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
	const std::string& filename,  // name of saved engine
	IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());


	//Optional - uncomment below lines to view network layer information
	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError << "Failure while parsing ONNX file" << std::endl;
	}

	// Build the engine
	builder->setMaxBatchSize(BATCH_SIZE);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setFp16Mode(true);
	builder->setInt8Mode(gArgs.runInInt8);

	if (gArgs.runInInt8)
	{
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder, gArgs.useDLACore);
	cout << "start building engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	cout << "build engine done" << endl;
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine 
	trtModelStream = engine->serialize();

	// save engine
	nvinfer1::IHostMemory* data = engine->serialize();
	std::ofstream file;
	file.open(filename, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());
	cout << "save engine file done" << endl;
	file.close();

	// then close everything down
	engine->destroy();
	network->destroy();
	builder->destroy();
}

inline int64_t volume(const nvinfer1::Dims& d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF: return 2;
	case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

//do inference all in one
void doInferenceFrieza(IHostMemory* trtModelStream, list<string> fileNames)
{
	//get engine
	assert(trtModelStream != nullptr);
	IRuntime* runtime = createInferRuntime(gLogger);
	nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
	assert(runtime != nullptr);
	if (gArgs.useDLACore >= 0)
	{
		runtime->setDLACore(gArgs.useDLACore);
	}
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), onnxPlugin);

	//get context
	assert(engine != nullptr);
	trtModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	//get buffers
	assert(engine->getNbBindings() == 4);
	void* buffers[4];
	std::vector<int64_t> bufferSize;
	int nbBindings = engine->getNbBindings();
	bufferSize.resize(nbBindings);

	for (int i = 0; i < nbBindings; ++i)
	{
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
		nvinfer1::DataType dtype = engine->getBindingDataType(i);
		int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
		bufferSize[i] = totalSize;
		cout << totalSize << endl;
		CHECK(cudaMalloc(&buffers[i], totalSize));
	}

	//get stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	//define inputImgs inputData outputDetections ...
	//vector<float> inputData;
	//inputData.reserve(DETECT_HEIGHT*DETECT_WIDTH*INPUT_CHANNEL*BATCH_SIZE);
	vector<cv::Mat> inputImgs;
	list<vector<DetectionRes>> outputs;
	int outSize1 = bufferSize[1] / sizeof(float);
	int outSize2 = bufferSize[2] / sizeof(float);
	int outSize3 = bufferSize[3] / sizeof(float);
	float* out1 = new float[outSize1];
	float* out2 = new float[outSize2];
	float* out3 = new float[outSize3];

	//start to do inference
	int index = 1,
		batchCount = 0;
	auto iter = fileNames.begin();
	for (unsigned int i = 0; i < fileNames.size(); ++i, ++iter)
	{
		const string& filename = *iter;
		cout << "process: " << filename << endl;

		cv::Mat img = cv::imread(filename);
		inputImgs.push_back(img);
		auto t_start_pre = std::chrono::high_resolution_clock::now();
		vector<float> curInput = prepareImage(img);
		auto t_end_pre = std::chrono::high_resolution_clock::now();
		float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
		std::cout << "prepare image take: " << total_pre << " ms." << endl;

		if (!curInput.data())
			continue;
		/*
		inputData.insert(inputData.end(), curInput.begin(), curInput.end());
		batchCount++;
		if (batchCount < BATCH_SIZE && i + 1 < fileNames.size())
			continue;
		*/

		// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
		CHECK(cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));

		// do inference
		auto t_start = std::chrono::high_resolution_clock::now();
		context->execute(BATCH_SIZE, buffers);
		auto t_end = std::chrono::high_resolution_clock::now();
		float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
		std::cout << "Inference take: " << total << " ms." << endl;

		CHECK(cudaMemcpyAsync(out1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
		CHECK(cudaMemcpyAsync(out2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream));
		CHECK(cudaMemcpyAsync(out3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		float* out = new float[outSize1 + outSize2 + outSize3];
		out = merge(out1, out2, out3, outSize1, outSize2, outSize3);

		// postprocess
		auto t_start_post = std::chrono::high_resolution_clock::now();
		auto boxes = postProcess(img, out);
		auto t_end_post = std::chrono::high_resolution_clock::now();
		float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
		std::cout << "Postprocess take: " << total_post << " ms." << endl;

		//print boxes
		for (int i = 0; i < boxes.size(); ++i)
		{
			cout << boxes[i].prob << ", " << boxes[i].x << ", " << boxes[i].y << ", " << boxes[i].w << ", " << boxes[i].h << endl;
		}

		outputs.push_back(boxes);

		cout << "\n" << endl;
	}

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[0]));
	CHECK(cudaFree(buffers[1]));
	CHECK(cudaFree(buffers[2]));
	CHECK(cudaFree(buffers[3]));

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// draw boxes
	int idx = 1;
	auto iterDet = outputs.begin();
	for (unsigned int i = 0; i < fileNames.size(); ++i, ++iterDet)
	{
		const vector<DetectionRes> &outputI = *iterDet;
		for (auto box : outputI)
		{
			int x = box.x,
				y = box.y,
				w = box.w,
				h = box.h;
			cv::Rect rect = { x, y, w, h };
			cv::rectangle(inputImgs[i], rect, cv::Scalar(255, 255, 0), 2);
		}
		stringstream ss;
		ss << idx;
		string index = ss.str();
		idx++;
		cv::imwrite("./result_" + index + ".jpg", inputImgs[i]);
		cout << "save result to: " << "./result_" + index + ".jpg" << endl;
	}

}

int main(int argc, char** argv)
{
	// read imgs list
	list<string> fileNames = readFileList(fileList);

	// create a TensorRT model from the onnx model and serialize it to a stream
	IHostMemory* trtModelStream{ nullptr };

	// create and load engine
	fstream existEngine;
	existEngine.open(engineFile, ios::in);
	if (existEngine)
	{
		readTrtFile(engineFile, trtModelStream);
		assert(trtModelStream != nullptr);
	}
	else
	{
		onnxToTRTModel(onnxFile, engineFile, trtModelStream);
		assert(trtModelStream != nullptr);
	}

	//do inference
	doInferenceFrieza(trtModelStream, fileNames);

	return 0;
}
