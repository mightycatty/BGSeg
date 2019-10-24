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
	void segImg(const cv::Mat& inputImg, cv::Mat& segResult, const bool staticFlag=false); // 

private:
	const std::string kModelName = "D:\\samba\\seg_demo_openvino19\\model_IR\\256_256\\fp32\\mobilenet_3.5B_with_mean-scale";
	VideoSmooth video_smoother_;
	ROIGenerator roi_generator_;
	VINOInference ie_;
	cv::Mat img_buffer_, img_buffer_roi_;
};
#endif