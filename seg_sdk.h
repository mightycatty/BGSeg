#pragma once
#pragma once
#pragma once
#ifndef _SEG_SDK_
#define _SEG_SDK_
#include "post_processing.h"
#include "roi_generator.h"
#include "ie_backend.h"
# include "utils.h"

// setBlob错误有release下缺少plugins.xml文件引起(vs重新生成会清理掉)
class SegSdk
{
public:
	SegSdk(std::string device="CPU", std::string cpu_threads="1");
	~SegSdk();
	bool segImg(cv::Mat& inputImg, cv::Mat& segResult); // 

private:
	VideoPost post_processor_;
	ROIGenerator roi_generator_;
	VINOInference ie_;
	cv::Mat img_roi_, roi_seg_result_;
};
#endif