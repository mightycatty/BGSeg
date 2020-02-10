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
	void Update(const std::vector<float>& relative_rect);
	void Update(cv::Mat& mask_result);
	void Reset(); // reset roi-generator to initial state


private:
	void ROISmooth(cv::Rect rectCurrent);
	cv::Rect getBoundingRectFromMask(const cv::Mat& binaryMask);

	cv::Rect ROI_rect_;
	
	float kUpdatePixelsMargineFraction_ = 0.03;
	float kUpdateFactor_ = 0.1;

	std::vector<size_t> or_shape_; // (width, height) 
	bool first_frame_flag_ = true; // set to false after the first call of Update()
	void ROICheck();
	std::chrono::high_resolution_clock::time_point last_time_stamp_ = std::chrono::high_resolution_clock::now();
	float reset_time_gap_ = 2000.; // reset roi when gap between sequential frames are above 2s
};

#endif // !_ROI_GENERATOR_
