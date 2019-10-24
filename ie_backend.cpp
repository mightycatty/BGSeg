#include "ie_backend.h"
#include "utils.h"
#include <time.h>
//#include "spdlog/spdlog.h"
//#include "spdlog/sinks/basic_file_sink.h"


VINOInference::VINOInference(std::string model_dir_name, std::string device, std::string cpu_threads)
{
	// logger
	//auto file_logger = spdlog::basic_logger_mt("ie_logger", "../logs/ie_logger.txt");
	//spdlog::set_default_logger(file_logger);
	
	err_msg_ = "";
	predict_time_ = 0.;
	/*try
	{*/
		// --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
		CNNNetReader networkReader;
		std::string model_xml = model_dir_name + ".xml";
		networkReader.ReadNetwork(model_xml);
		std::string bin_file_name = model_dir_name + ".bin";
		networkReader.ReadWeights(bin_file_name);
		CNNNetwork network = networkReader.getNetwork();
		network.setBatchSize(1);
		// --------------------------- 2. Configure input & output ---------------------------------------------
		// --------------------------- Prepare input blobs -----------------------------------------------------
		InputsDataMap inputInfo(network.getInputsInfo());
		auto inputInfoItem = *inputInfo.begin(); // merely single input in the model
		auto inputData = inputInfoItem.second;
		input_name_ = inputInfoItem.first;
		inputData->setPrecision(Precision::U8);
		inputData->getPreProcess().setColorFormat(ColorFormat::RGB);
		inputData->setLayout(Layout::NCHW);
		//inputData->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);


		// --------------------------- Prepare output blobs ----------------------------------------------------
		OutputsDataMap outputInfo(network.getOutputsInfo());
		auto outputInfoItem = *outputInfo.begin();
		output_name_ = outputInfoItem.first;
		auto outputData = outputInfoItem.second;
		outputData->setPrecision(Precision::FP32);
		outputData->setLayout(Layout::NCHW);
		// --------------------------- resize network ---------------------------------------------------------
		//auto input_shapes = network.getInputShapes();
		//std::string input_name;
		//SizeVector input_shape;
		//std::tie(input_name, input_shape) = *input_shapes.begin(); // let's consider first input only
		//input_shape[0] = 1; // set batch size to the first input dimension
		//input_shape[2] = config_.height; // changes input height to the image one
		//input_shape[3] = config_.width; // changes input width to the image one
		//input_shapes[input_name] = input_shape;
		//network.reshape(input_shapes);
		// --------------------------- 3. initialize IECore && Loading model to specific device ------------------------------------------
		Core ie;
		ExecutableNetwork executable_network;
		auto extension_ptr = make_so_pointer<IExtension>("cpu_extension_avx2.dll");
		ie.AddExtension(extension_ptr, "CPU");
		std::vector<std::string> available_devices = ie.GetAvailableDevices();
		bool GPU_found = (std::find(available_devices.begin(), available_devices.end(), "GPU") != available_devices.end());
		bool cpu_flag = false;
		// TODO: device 
		if (true)
		{
			std::map<std::string, std::string> ieConfig = {
				{InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, cpu_threads},
				{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES},
				{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "1"}
			};
			executable_network = ie.LoadNetwork(network, "CPU", ieConfig);
		}
		else
		{
			executable_network = ie.LoadNetwork(network, "GPU");
		}

		// --------------------------- 5. Create infer request -------------------------------------------------
		//infer_request_ = executable_network.CreateInferRequest(); // the end of IE initialization

		m_async_infer_request_curr = executable_network.CreateInferRequestPtr();
		m_async_infer_request_next = executable_network.CreateInferRequestPtr();

		/* it's enough just to set image info input (if used in the model) only once */

	//}
	//catch (InferenceEngineException e) {
	//	//spdlog::error("error with OPENVINO_IE initialization");
	//	err_msg_ = "error with initializing openvino IE";
	//}
}



VINOInference::~VINOInference()
{
}


// unneccesary when embeded in IE
void VINOInference::Preprocessing(const cv::Mat& kInputImg, cv::Mat& output_img, int resize_width)
{
	ResizeWithPadding(kInputImg, output_img, resize_width);
	return ;
}


// trim padding and restore to the original shape
void VINOInference::PostProcessing(const cv::Mat& inputImg, cv::Mat& mask)
{
	int or_height = inputImg.size().height;
	int or_width = inputImg.size().width;

	float ratio = float(std::min(or_height, or_width)) / float(std::max(or_height, or_width));
	
	float mask_width = mask.size().width;
	cv::Rect roi;
	if (or_width >= or_height)
	{
		roi.x = 0;
		roi.height = mask_width * ratio;
		roi.y = (mask_width - roi.height) / 2;
		roi.width = mask_width;
	}
	else
	{
		roi.y = 0;
		roi.height = mask_width;
		roi.width = mask_width * ratio;
		roi.x = (mask_width - roi.width) / 2;
	}
	resize(mask(roi), mask, cv::Size(or_width, or_height));
}


void VINOInference::getOutput(cv::Mat& matResult, InferRequest::Ptr& inferRequest)
{
	//output_blob_ = infer_request_.GetBlob(output_name_);
	output_blob_ = inferRequest->GetBlob(output_name_);
	auto output_data = output_blob_->buffer().as<float*>();

	size_t C, H, W;
	C = 2; // binary segmentation with 2 classes 
	H = input_height_;
	W = input_width;
	size_t image_stride = W * H * C;
	///** Iterating over each pixel **/
	matResult = cv::Mat(cv::Size(W, H), CV_8UC1);
	for (size_t w = 0; w < W; ++w) {
		for (size_t h = 0; h < H; ++h) {
			/* number of channels = 1 means that the output is already ArgMax'ed */
			//if (C == 1) {
			//	outArrayClasses[h][w] = static_cast<float>(output_data[W * h + w]);
			//}
			//else {
				/** Iterating over each class probability **/
				//for (int ch = 0; ch < C; ++ch) {
				//	auto data = output_data[W * H * ch + W * h + w];
				//	if (data > outArrayProb[h][w]) {
				//		outArrayClasses[h][w] = static_cast<float>(ch);
				//		outArrayProb[h][w] = data;
				//	}
				//}
				if (output_data[W * h + w] > output_data[W * H * 1 + W * h + w])
				{
					matResult.at<uchar>(h, w) = 0;
				}
				else
				{
					matResult.at<uchar>(h, w) = 1;
				}
			//}
		}
	}
	
	//for (int i = 0; i < matResult.rows; ++i)
	//	for (int j = 0; j < matResult.cols; ++j)
	//		matResult.at<double>(i, j) = outArrayClasses.at(i).at(j);
	//matResult.convertTo(matResult, CV_8U); 
	return ;
}


void VINOInference::Predict(const cv::Mat& imageData, cv::Mat& result)
{
	//auto start = clock();
	Preprocessing(imageData, img_prepro_);
	img_blob_ = wrapMat2Blob(img_prepro_);
	//img_prepro_.convertTo(img_prepro_, CV_8UC1);

	infer_request_.SetBlob(input_name_, img_blob_);
	infer_request_.Infer();

	//predict_time = clock() - start;
	//spdlog::info("Inference time:{:2.4f}", duration);
	getOutput(result, m_async_infer_request_curr);
	PostProcessing(imageData, result); // restore to original image size
	return ;
}

bool VINOInference::PredictAsync(const cv::Mat& imageData, cv::Mat& result)
{
	static bool bFirst = true;
	static cv::Mat curr_frame;  
	if (bFirst) {
		bFirst = false;
		curr_frame = imageData;
		return false;
	}

	Preprocessing(curr_frame, img_prepro_);
	img_blob_ = wrapMat2Blob(img_prepro_);

	m_async_infer_request_curr->SetBlob(input_name_, img_blob_);
	m_async_infer_request_curr->StartAsync();

	if (OK == m_async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
		getOutput(result, m_async_infer_request_curr);
		PostProcessing(curr_frame, result); // restore to original image size
	}

	curr_frame = imageData;
	m_async_infer_request_curr.swap(m_async_infer_request_next);

	return true;
}
