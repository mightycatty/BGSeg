#pragma once
#pragma once
#ifndef _POST_PROCESSOR_
#define _POST_PROCESSOR_
#include <opencv2/opencv.hpp>
#include <vector>


void SizeNorm(cv::Mat& matItem, int maxDim=320);
void ContourRefine(cv::Mat& matItem);
void EdgeSmooth(cv::Mat& binary_mask, const int kIteration=2);

class VideoSmooth
{
public:
	VideoSmooth();
	~VideoSmooth();
	void Reset();
	void Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool seq_fusion_flag=true);


private:
	void MaskSmooth(cv::Mat& prev_mask, cv::Mat& current_mask, cv::Mat& result);
	void SeqMaskFusion(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& fusion_result);


	cv::Mat prev_mask_;
	const float kSmoothDegree_ = 0.90;
	const int kSmoothLen_ = 5;
	std::vector<cv::Mat> img_seq_buffer_;
	std::vector<cv::Mat> mask_seq_buffer_;
	// intermidiate 
	cv::Mat mask_seq_sum_;
	cv::Mat mask_inter_, mask_un_;
	cv::Mat img_seq_sum_;



};
#endif // 
