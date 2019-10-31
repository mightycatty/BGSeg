#pragma once
#pragma once
#ifndef _POST_PROCESSOR_
#define _POST_PROCESSOR_
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>


void SizeNorm(cv::Mat& matItem, int maxDim=320);
void EdgeSmooth(cv::Mat& binary_mask, const int kIteration=2);

class VideoPost
{
public:
	VideoPost();
	~VideoPost();
	void Reset();
	void Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool seq_fusion_flag=true);
	void ContourRefine(cv::Mat& matItem);

	std::vector<cv::Point> largest_cnt_;

private:
	void MaskSmooth(cv::Mat& current_mask, cv::Mat& result);
	void SeqMaskFusion(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& fusion_result);

	cv::Mat prev_mask_;
	const float kSmoothDegree_ = 0.85;
	const int kSmoothLen_ = 3;
	std::vector<cv::Mat> img_seq_buffer_;
	std::vector<cv::Mat> mask_seq_buffer_;
	// intermidiate 
	cv::Mat mask_seq_sum_;
	cv::Mat mask_inter_, mask_un_;
	cv::Mat img_seq_sum_;
	cv::Mat fusion_mask_;
	std::chrono::high_resolution_clock::time_point last_time_stamp_ = std::chrono::high_resolution_clock::now();
	float reset_time_gap_ = 200.;
};
#endif // 
