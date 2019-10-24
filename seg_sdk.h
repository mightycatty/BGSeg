#pragma once
#pragma once
#pragma once
#ifndef _SEG_SDK_
#define _SEG_SDK_
#include "post_processing.h"
#include "roi_generator.h"
#include "ie_backend.h"
# include "utils.h"

// setBlob������release��ȱ��plugins.xml�ļ�����(vs�������ɻ������)
class SegSdk
{
public:
	SegSdk(std::string device="CPU", std::string cpu_threads="1");
	~SegSdk();
	bool segImg(cv::Mat& inputImg, cv::Mat& segResult, const bool staticFlag=false); // 

private:
	const std::string kModelName = "../model_IR/600M_512_fp16/600M";
	VideoSmooth video_smoother_;
	ROIGenerator roi_generator_;
	VINOInference ie_;
	cv::Mat img_buffer_roi_;
};
#endif