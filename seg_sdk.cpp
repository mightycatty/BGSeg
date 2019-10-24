#include "seg_sdk.h"

SegSdk::SegSdk(std::string device, std::string cpu_threads)
	: ie_(kModelName, device, cpu_threads)
{
}

SegSdk::~SegSdk()
{
	//
}

void SegSdk::segImg(const cv::Mat& inputImg, cv::Mat& segResult, bool staticFlag)
{
	inputImg.copyTo(img_buffer_);
	cv::imshow("original", img_buffer_);
	roi_generator_.getROIImage(img_buffer_, img_buffer_roi_);
	cv::imshow("roi", img_buffer_roi_);
	ie_.Predict(img_buffer_roi_, segResult);
	cv::imshow("raw_cnn", segResult * 255);
	ContourRefine(segResult);
	cv::imshow("cnt", segResult * 255);
	roi_generator_.restoreFromROI(segResult, segResult);
	cv::imshow("before_smooth", segResult * 255);
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
	cv::imshow("aftersmooth", segResult * 255);
}