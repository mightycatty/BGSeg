#pragma once
#ifndef _ROI_GENERATOR_
#define _ROI_GENERATOR_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// TODO：增加基于指数平滑的平滑策略
class ROIGenerator
{
public:
	ROIGenerator();
	~ROIGenerator();

	void getROIImage(cv::Mat& img_src, cv::Mat& img_roi);
	void restoreFromROI(cv::Mat& mask_src, cv::Mat& mask_restored);
	void Reset(); // reset roi-generator to initial state
	

private:
	void ROISmooth(cv::Rect rectCurrent);
	cv::Rect getBoundingRectFromMask(const cv::Mat& binaryMask);
	void Update(const cv::Mat& mask);

	const float kAlpha_ = 0.2;
	cv::Rect ROI_rect_;
	/*float s[4] = { 0, 0, 0, 0 };
	float b[4] = { 0, 0, 0, 0 };
	*/
	int kUpdatePixelsMargine_ = 5;
	const float kRoiLeastSizeFraction_ = 0.2; // for SME smooth
	std::vector<size_t> or_shape_; // (width, height) 
	bool first_frame_flag_ = true; // set to false after the first call of Update()
	void ROICheck();
};

#endif // !_ROI_GENERATOR_
