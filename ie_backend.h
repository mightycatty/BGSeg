#ifndef __VINOIE_H_
#define __VINOIE_H_
#pragma warning(disable:4251)  //needs to have dll-interface to be used by clients of class 
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <ie_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>
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
	VINOInference(std::string device="AUTO", std::string cpu_threads="1");
	~VINOInference();

	bool Predict(const cv::Mat& orgimg, cv::Mat& result, bool restore_shape);
	//bool PredictAsync(const cv::Mat& imageData, cv::Mat& result, bool restore_shape = false);
	void RestoreShape(const cv::Mat& kInputImg, cv::Mat& mask);

	std::string err_msg_;
	float predict_time_;
	const std::string kModelSmall = "../model_IR/600M/320_fp16/scale_600M";//"../model_IR/256_256/fp16/mobilenet_3.5B_with_mean-scale";
	const std::string kModelMiddle = "../model_IR/3B/256_256/fp16/mobilenet_3.5B_with_mean-scale";

	int input_width_ = 320;

private:

	//InferRequest::Ptr m_async_infer_request_curr_;
	//InferRequest::Ptr m_async_infer_request_next_;
	InferRequest infer_request_;

	std::string input_name_;
	std::string output_name_;
	Blob::Ptr img_blob_, output_blob_;
	cv::Mat img_prepro_;

	void Preprocessing(const cv::Mat& kInputImg, cv::Mat& output_img, int resize_width=256); //which is not required when embeded in openvinoIE
	void getOutput(cv::Mat&, InferRequest& inferRequest);
	InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat);

};

#endif //__VINOIE_H_

//TODO: device query
//TODO: model distribution base on image resolution