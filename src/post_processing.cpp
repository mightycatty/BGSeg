//#include <Eigen>
//#include <opencv2/core/eigen.hpp>
#include "post_processing.h"
#include "utils.h"
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


using namespace std;
//using namespace Eigen;


std::vector<float> VideoPost::ContourRefine(cv::Mat& matItem)
{
	// refine mask by eliminating small patch of foreground
	static std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> largest_cnt;
	static cv::Rect rect_current;
	static cv::Mat mat_binary;
	cv::threshold(matItem, mat_binary, 0.5, 1, cv::THRESH_BINARY);
	mat_binary.convertTo(mat_binary, CV_8UC1);
	findContours(mat_binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() >= 1)
	{
		std::sort(contours.begin(), contours.end(), ContoursSortFun);
		cv::Mat matItem_cnt = cv::Mat::zeros(matItem.size(), CV_8UC1);
		drawContours(matItem_cnt, contours, 0, cv::Scalar(1.), -1);
		matItem_cnt.convertTo(matItem_cnt, CV_32FC1);
		cv::multiply(matItem, matItem_cnt, matItem);
		largest_cnt = contours[0];

		rect_current = cv::boundingRect(largest_cnt);
		relative_rect_[0] = rect_current.x / float(matItem.size().width);
		relative_rect_[2] = rect_current.width / float(matItem.size().width);
		relative_rect_[1] = rect_current.y / float(matItem.size().height);
		relative_rect_[3] = rect_current.height / float(matItem.size().height);
	}
	else
	{
		relative_rect_ = {0., 0., 0., 0.};
	}

	return relative_rect_;
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
	static float smooth_degree = 0.80;
	if (!(prev_mask_.empty()))
	{
		if (!fusion_mask_.empty())
		{
			fusion_mask_ = fusion_mask_ * smooth_degree + (1. - fusion_mask_) * (1 - smooth_degree);
			cv::multiply((1. - fusion_mask_), prev_mask_, a);
			cv::multiply(fusion_mask_, current_mask, b);
			cv::add(a, b, result);
		}
		else
		{
			//result = float(1 - kSmoothDegree_) * prev_mask + float(kSmoothDegree_) * current_mask;
			cv::addWeighted(current_mask, (smooth_degree), prev_mask_, (1- smooth_degree), 0., result);
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
	static cv::Mat img_gray;
	static float acc_factor = 0.8;
	static double min, max;
	cv::cvtColor(new_img, img_gray, cv::COLOR_BGR2GRAY);
	if (not prev_img_.empty())
	{
		cv::absdiff(prev_img_, img_gray, img_diff);
		//img_diff.convertTo(img_diff, CV_8UC1);
		//cv::multiply(img_diff, new_mask, img_diff);
		cv::minMaxLoc(img_diff, &min, &max);
		cv::threshold(img_diff, fusion_mask_, 0.2*max, 1, cv::THRESH_BINARY);
		int erosion_size = 5;
		cv::Mat kernel = getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		cv::dilate(fusion_mask_, fusion_mask_, kernel);
		fusion_mask_.convertTo(fusion_mask_, CV_32FC1);
		//cv::imshow("fusion_mask", fusion_mask_*255);

		prev_img_ = acc_factor *  prev_img_ + (1 - acc_factor) * img_gray;
		//cv::imshow("avg_img", prev_img_);
		//cv::waitKey(1);
	}
	else
	{
		img_gray.copyTo(prev_img_);
	}
}


void VideoPost::Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool compact_mode)
{
	static cv::Mat new_img_down;
	static double duration_ms;
	duration_ms = std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - last_time_stamp_).count();
	last_time_stamp_ = std::chrono::high_resolution_clock::now();
	if (duration_ms > reset_time_gap_) { Reset(); }  // if time gap between two frame is larger than threshold ,reset postprocessor
	if (!compact_mode)
	{
		cv::resize(new_img, new_img_down, cv::Size(post_processing_size_, post_processing_size_), cv::INTER_NEAREST);
		GetMotionMask(new_img_down);
		if (!fusion_mask_.empty())
		{
			cv::resize(fusion_mask_, fusion_mask_, new_img.size());
		}
		MaskSmooth(result, result);
		ContourRefine(result);
		// edge refine
		for (int i = 0; i < 2; i++)
		{
			cv::medianBlur(result, result, 5);
		}
		int erode_size = 1;
		static cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * erode_size + 1, 2 * erode_size + 1),
			cv::Point(erode_size, erode_size));
		/// Apply the dilation operation
		cv::erode(result, result, element);
	}
	else
	{
		new_mask.copyTo(result);
		ContourRefine(result);
	}
}

