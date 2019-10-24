#ifndef __VINOIE_H_
#define __VINOIE_H_
#pragma warning(disable:4251)  //needs to have dll-interface to be used by clients of class 
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <ie_plugin_config.hpp>
#include <ie_plugin_ptr.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>
#include "config.h"

/**********************************************************************/


using std::queue;
using std::string;
using std::vector;
using namespace InferenceEngine::details;
using namespace InferenceEngine;


class VINOInference
{
public:
	VINOInference(std::string model_dir_name, std::string device="AUTO", std::string cpu_threads="1");
	~VINOInference();

	void Predict(const cv::Mat& orgimg, cv::Mat& result);
	bool PredictAsync(const cv::Mat& imageData, cv::Mat& result);

	std::string err_msg_;
	float predict_time_;
	int input_height_ = 256;
	int input_width = 256;

private:

	InferRequest infer_request_;
	InferRequest::Ptr m_async_infer_request_curr;
	InferRequest::Ptr m_async_infer_request_next;

	std::string input_name_;
	std::string output_name_;
	Blob::Ptr img_blob_, output_blob_;
	cv::Mat img_prepro_;
	void Preprocessing(const cv::Mat& kInputImg, cv::Mat& output_img, int resize_width=256); //which is not required when embeded in openvinoIE
	void getOutput(cv::Mat&, InferRequest::Ptr& inferRequest);
	void PostProcessing(const cv::Mat& kInputImg, cv::Mat& mask);
};

#endif //__VINOIE_H_

//TODO: device query
//TODO: model distribution base on image resolution