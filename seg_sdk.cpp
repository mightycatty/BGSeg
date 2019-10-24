#include "seg_sdk.h"

SegSdk::SegSdk(std::string device, std::string cpu_threads)
	: ie_(kModelName, device, cpu_threads)
{
}

SegSdk::~SegSdk()
{
	//
}

bool SegSdk::segImg(const cv::Mat& inputImg, cv::Mat& segResult, bool staticFlag)
{
	inputImg.copyTo(img_buffer_);
	roi_generator_.getROIImage(img_buffer_, img_buffer_roi_);
	if (!ie_.PredictAsync(img_buffer_roi_, segResult)) {
		return false;
	}
	ContourRefine(segResult);
	roi_generator_.restoreFromROI(segResult, segResult);
	// size norm for processing downstream
	if (!staticFlag)
	{
		int orHeight = inputImg.size().height;
		int orWidth = inputImg.size().width;
		SizeNorm(img_buffer_, 320);
		SizeNorm(segResult, 320);
		video_smoother_.Process(img_buffer_, segResult, segResult);
		cv::resize(segResult, segResult, cv::Size(orWidth, orHeight));
	}
	else
	{
		video_smoother_.Reset();
	}
}