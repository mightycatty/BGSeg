#pragma once
#pragma once
#pragma once
#ifndef _SEG_SDK_
#define _SEG_SDK_
#include "post_processing.h"
#include "roi_generator.h"
#include "ie_backend.h"
# include "utils.h"


class SegSdk
{
public:
	SegSdk(int model_index = 1, bool force_cpu_mode = false, std::string cpu_threads = "1", std::string model_folder="../models/"); // 0:compact 1: modest, 2: heavy
	~SegSdk();
	bool segImg(cv::Mat& inputImg, cv::Mat& segResult, std::string cvt_color="RGB"); // 

private:
	VideoPost post_processor_;
	ROIGenerator roi_generator_;
	VINOInference ie_;
	bool compact_mode_ = false;
	cv::Size seg_shape_ = cv::Size(480, 360); // segmentation input resolution: 360p
};
#endif