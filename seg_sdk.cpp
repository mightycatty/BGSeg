#include "seg_sdk.h"

SegSdk::SegSdk(std::string device, std::string cpu_threads)
	: ie_(device, cpu_threads)
{
	cv::setNumThreads(0);
}

SegSdk::~SegSdk()
{
	//
}

bool SegSdk::segImg(cv::Mat& inputImg, cv::Mat& segResult)
{
	roi_generator_.getROIImage(inputImg, img_roi_);
	/*cv::imshow("roi", img_roi_);
	cv::imshow("cam", inputImg);*/
	if (!ie_.Predict(img_roi_, roi_seg_result_, false)) {return false;}
	//cv::imshow("raw", roi_seg_result_);
	ie_.RestoreShape(img_roi_, roi_seg_result_);
	// post
   	roi_generator_.restoreFromROI(roi_seg_result_, segResult);
	post_processor_.Process(inputImg, segResult, segResult, false);
	post_processor_.ContourRefine(segResult);
	//cv::imshow("after-post", segResult);
	roi_generator_.Update(post_processor_.largest_cnt_);
	return true;
}