#include "seg_sdk.h"

SegSdk::SegSdk(std::string device, std::string cpu_threads)
	: ie_(kModelName, device, cpu_threads)
{
}

SegSdk::~SegSdk()
{
	//
}

bool SegSdk::segImg(cv::Mat& inputImg, cv::Mat& segResult, bool staticFlag)
{
	//int orWidth = inputImg.size().width;
	//int orHeight = inputImg.size().height;
	//cv::imshow("raw", inputImg);
	roi_generator_.getROIImage(inputImg, img_buffer_roi_);
	//cv::imshow("roi", img_buffer_roi_);
	if (!ie_.PredictAsync(img_buffer_roi_, segResult)) {
		return false;
	}
	//cv::imshow("cnn_raw", segResult);
	ContourRefine(segResult);
	//cv::imshow("after_cnt", segResult);
	roi_generator_.restoreFromROI(segResult, segResult);
	// size norm for processing downstream
	if (!staticFlag)
	{
		if (segResult.size() != inputImg.size()) {
			cv::resize(segResult, segResult, inputImg.size());
		}
		//SizeNorm(inputImg, 320);
		//SizeNorm(segResult, 320);
		video_smoother_.Process(inputImg, segResult, segResult, false);
		//cv::resize(segResult, segResult, cv::Size(orWidth, orHeight));
	}
	else
	{
		video_smoother_.Reset();
	}
	//cv::imshow("final", segResult);
	//cv::waitKey(1);
}