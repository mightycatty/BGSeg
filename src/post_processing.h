#pragma once
#pragma once
#ifndef _POST_PROCESSOR_
#define _POST_PROCESSOR_
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>


void SizeNorm(cv::Mat& matItem, int maxDim=320);

class VideoPost
{
public:
	VideoPost();
	~VideoPost();
	void Reset();
	void Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool compact_mode=true);
	std::vector<float> ContourRefine(cv::Mat& matItem);
	int post_processing_size_ = 144;

	std::vector<float> relative_rect_ = { 0., 0., 0., 0. };

private:
	void MaskSmooth(cv::Mat& current_mask, cv::Mat& result);
	void GetMotionMask(const cv::Mat& new_img);
	cv::Mat prev_img_, fusion_mask_, prev_mask_;

	std::chrono::high_resolution_clock::time_point last_time_stamp_ = std::chrono::high_resolution_clock::now();
	float reset_time_gap_ = 200.;

};
#endif // 
