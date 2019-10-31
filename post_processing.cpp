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
	cv::threshold(matItem, mat_binary, 0.5, 1, cv::THRESH_BINARY);
	mat_binary.convertTo(mat_binary, CV_8UC1);
	findContours(mat_binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), ContoursSortFun);
	cv::Mat matItem_cnt = cv::Mat::zeros(matItem.size(), CV_8UC1);
	drawContours(matItem_cnt, contours, 0, cv::Scalar(1.), -1);
	matItem_cnt.convertTo(matItem_cnt, CV_32FC1);
	cv::multiply(matItem, matItem_cnt, matItem);
	largest_cnt_ = contours[0];
}

static void EdgeSmooth(cv::Mat& binary_mask, const int kIteration)
{
	cv::GaussianBlur(binary_mask, binary_mask, cv::Size(3, 3), 0);
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
	img_seq_sum_.release();
	mask_seq_sum_.release();
	img_seq_buffer_.clear();
	mask_seq_buffer_.clear();
}


void VideoPost::MaskSmooth(cv::Mat& current_mask, cv::Mat& result)
{
	if (!(prev_mask_.empty()))
	{
		if (!fusion_mask_.empty())
		{
			fusion_mask_ = fusion_mask_ * 0.85 + (1 - fusion_mask_) * 0.1;
			result = (1 - fusion_mask_) * prev_mask_ + fusion_mask_ * current_mask;
		}
		else
		{
			//result = float(1 - kSmoothDegree_) * prev_mask + float(kSmoothDegree_) * current_mask;
			cv::addWeighted(current_mask, kSmoothDegree_, prev_mask_, 1.0 - kSmoothDegree_, 0., result);
		}
	}
	else
	{
		result = current_mask;
	}
	result.copyTo(prev_mask_);
}


void VideoPost::SeqMaskFusion(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& fusion_result)
{
	int height = new_img.size().height;
	int width = new_img.size().width;
	// img & mask proprocessing
	cv::pyrDown(new_img, new_img);
	new_img.convertTo(new_img, CV_32FC3);
	img_seq_buffer_.push_back(new_img.clone()); // matÇ³¿½±´£¬ÐèÒªclone
	mask_seq_buffer_.push_back(new_mask.clone());
	// sum up mask
	if (mask_seq_sum_.empty()) mask_seq_sum_ = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	mask_seq_sum_ += new_mask;
	if (img_seq_sum_.empty()) img_seq_sum_ = cv::Mat::zeros(cv::Size(width / 2, height / 2), CV_32FC3);
	img_seq_sum_ += new_img;
	if (int(img_seq_buffer_.size()) == kSmoothLen_)
	{
		// ===============get the intersection and union of seq masks=====================
		cv::threshold(mask_seq_sum_, mask_inter_, double(kSmoothLen_-1), 1, cv::THRESH_BINARY);
		cv::threshold(mask_seq_sum_, mask_un_, 0, 1, cv::THRESH_BINARY);
		// ================ std for img seq ===================
		cv::Mat img_mean = img_seq_sum_ / float(kSmoothLen_);
		cv::Mat img_square_sum = cv::Mat::zeros(cv::Size(width / 2, height / 2), CV_32FC3);
		for (int i=0; i < kSmoothLen_ ; ++i)
		{
			auto img_item = img_seq_buffer_[i].clone();
			img_item -= img_mean;
			cv::multiply(img_item, img_item, img_item);
			img_square_sum += img_item;
		}
		cv::Mat img_std = img_square_sum / float(kSmoothLen_ - 1);
		cv::sqrt(img_std, img_std);
		// =================== get fusion mask out of img std(img dynamic) ================================
		cv::transform(img_std, fusion_mask_, cv::Matx13f(1, 1, 1));
		double min, max;
		cv::minMaxLoc(fusion_mask_, &min, &max);
		int max_threshold = 50; //**TODO**: hardcode threhold, might lead to performance hit under different environment
		if (max < max_threshold)
		{
			mask_inter_.copyTo(fusion_result);
		}
		else
		{
			fusion_mask_ /= max;
			fusion_mask_ *= 255.;
			fusion_mask_.convertTo(fusion_mask_, CV_8UC1);
			/*cv::namedWindow("img_std");
			cv::imshow("img_std", fusion_mask);*/
			cv::threshold(fusion_mask_, fusion_mask_, 0, 1, cv::THRESH_BINARY+cv::THRESH_OTSU);
			dilate(fusion_mask_, fusion_mask_, cv::Mat(), cv::Point(-1, -1), 3); // 3times
			pyrUp(fusion_mask_, fusion_mask_);
			/*cv::namedWindow("img_std_binary");
			cv::imshow("img_std_binary", fusion_mask * 255);*/
			//fusion_result = fusion_mask * new_mask + (1 - fusion_mask) * mask_inter_;
			cv::copyTo(new_mask, fusion_result, fusion_mask_); // directly copy pixel value with mask, which yields identic result as above
			cv::copyTo(mask_inter_, fusion_result, 1 - fusion_mask_);
		}
		// ===========================fusion===============================
		// erase the oldest img/mask pair
		mask_seq_sum_ -= mask_seq_buffer_[0];
		img_seq_sum_ -= img_seq_buffer_[0];
		img_seq_buffer_.erase(img_seq_buffer_.begin());
		mask_seq_buffer_.erase(mask_seq_buffer_.begin());
	}
}


void VideoPost::Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result, bool seq_fusion_flag)
{
	static double duration_ms;
	duration_ms = std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - last_time_stamp_).count();
	last_time_stamp_ = std::chrono::high_resolution_clock::now();
	if (duration_ms > reset_time_gap_) { Reset(); }  // if time gap between two frame is larger than threshold ,reset postprocessor
	if (seq_fusion_flag)
	{
		SeqMaskFusion(new_img, new_mask, result);
	}
	else
	{
		result = new_mask;
	}
	MaskSmooth(result, result);
}