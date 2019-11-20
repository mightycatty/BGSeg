//#include <Eigen>
//#include <opencv2/core/eigen.hpp>
#include "post_processing.h"
#include "utils.h"
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


using namespace std;
//using namespace Eigen;


static void SizeNorm(cv::Mat& matItem, int maxDim)
{
	auto height = matItem.size().height;
	auto width = matItem.size().width;
	int shortDim = (int)round(maxDim / 256. * 144.);
	cv::Size reSize = (height > width) ? cv::Size(shortDim, 320) : cv::Size(320, shortDim);
	resize(matItem, matItem, reSize);
}

void VideoPost::ContourRefine(cv::Mat& matItem)
{
	// refine mask by eliminating small patch of foreground
	static std::vector<std::vector<cv::Point> > contours;
	static cv::Mat mat_binary;
	cv::threshold(matItem, mat_binary, 0.1, 1, cv::THRESH_BINARY);
	mat_binary.convertTo(mat_binary, CV_8UC1);
	findContours(mat_binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), ContoursSortFun);
	cv::Mat matItem_cnt = cv::Mat::zeros(matItem.size(), CV_8UC1);
	drawContours(matItem_cnt, contours, 0, cv::Scalar(1.), -1);
	matItem_cnt.convertTo(matItem_cnt, CV_32FC1);
	cv::multiply(matItem, matItem_cnt, matItem);
	largest_cnt_ = contours[0];
}


VideoPost::VideoPost()
{
	//
}


VideoPost::~VideoPost()
{
	// 
}


void VideoPost::Reset()
{
	prev_mask_.release();
	fusion_mask_.release();
	prev_img_.release();
}


void VideoPost::MaskSmooth(cv::Mat& current_mask, cv::Mat& result)
{
	static cv::Mat a, b;
	if (!(prev_mask_.empty()))
	{
		if (!fusion_mask_.empty())
		{
			fusion_mask_ = fusion_mask_ * 0.95 + (1. - fusion_mask_) * 0.1;
			cv::multiply((1. - fusion_mask_), prev_mask_, a);
			cv::multiply(fusion_mask_, current_mask, b);
			cv::add(a, b, result);
		}
		else
		{
			//result = float(1 - kSmoothDegree_) * prev_mask + float(kSmoothDegree_) * current_mask;
			cv::addWeighted(current_mask, 0.95, prev_mask_, 0.05, 0., result);
		}
	}
	else
	{
		result = current_mask;
	}
	result.copyTo(prev_mask_);
}


void VideoPost::GetMotionMask(const cv::Mat& new_img)
{
	static cv::Mat img_diff;
	if (not prev_img_.empty())
	{
		cv::absdiff(prev_img_, new_img, img_diff);
		cv::cvtColor(img_diff, fusion_mask_, cv::COLOR_BGR2GRAY);
		cv::threshold(fusion_mask_, fusion_mask_, 20, 1, cv::THRESH_BINARY);
		int erosion_size = 11;
		cv::Mat kernel = getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		cv::dilate(fusion_mask_, fusion_mask_, kernel);
		fusion_mask_.convertTo(fusion_mask_, CV_32FC1);
	}
	new_img.copyTo(prev_img_);
}


void VideoPost::Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool seq_fusion_flag)
{
	static double duration_ms;
	duration_ms = std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - last_time_stamp_).count();
	last_time_stamp_ = std::chrono::high_resolution_clock::now();
	if (duration_ms > reset_time_gap_) { Reset(); }  // if time gap between two frame is larger than threshold ,reset postprocessor
	GetMotionMask(new_img);
	MaskSmooth(result, result);
}