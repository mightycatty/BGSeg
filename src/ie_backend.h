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
	VINOInference(int model_index=1, std::string cpu_threads="1", bool force_cpu_mode=FALSE, std::string model_folder="../models/");
	~VINOInference();

	bool Predict(const cv::Mat& orgimg, cv::Mat& result);
	//bool PredictAsync(const cv::Mat& imageData, cv::Mat& result, bool restore_shape = false);

	std::string err_msg_;
	float predict_time_;
	int input_shape_;


private:

	//InferRequest::Ptr m_async_infer_request_curr_;
	//InferRequest::Ptr m_async_infer_request_next_;
	InferRequest infer_request_;

	std::string input_name_;
	std::string output_name_;
	Blob::Ptr img_blob_, output_blob_;

	void getOutput(cv::Mat&, InferRequest& inferRequest); // rewrite this for your own network
	InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat);

};

#endif //__VINOIE_H_

//TODO: device query
//TODO: model distribution base on image resolution