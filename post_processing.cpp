//#include <Eigen>
//#include <opencv2/core/eigen.hpp>
#include "post_processing.h"
#include "utils.h"


using namespace std;
//using namespace Eigen;


void SizeNorm(cv::Mat& matItem, int maxDim)
{
	auto height = matItem.size().height;
	auto width = matItem.size().width;
	int shortDim = (int)round(maxDim / 256. * 144.);
	cv::Size reSize = (height > width) ? cv::Size(shortDim, 320) : cv::Size(320, shortDim);
	resize(matItem, matItem, reSize);
}

void ContourRefine(cv::Mat& matItem)
{
	// refine mask by eliminating small patch of foreground
	std::vector<std::vector<cv::Point> > contours;
	findContours(matItem.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), ContoursSortFun);
	cv::Mat matItem_cnt = cv::Mat::zeros(matItem.size(), CV_8U);
	drawContours(matItem_cnt, contours, 0, cv::Scalar(1), -1);
	cv::bitwise_and(matItem, matItem_cnt, matItem);
}

void EdgeSmooth(cv::Mat& binary_mask, const int kIteration)
{
	cv::GaussianBlur(binary_mask, binary_mask, cv::Size(3, 3), 0);
}

VideoSmooth::VideoSmooth()
{
	//
}


VideoSmooth::~VideoSmooth()
{
	// 
}


void VideoSmooth::Reset()
{
	prev_mask_.release();
	img_seq_sum_.release();
	mask_seq_sum_.release();
	img_seq_buffer_.clear();
	mask_seq_buffer_.clear();
}


void VideoSmooth::MaskSmooth(cv::Mat& prev_mask, cv::Mat& current_mask, cv::Mat& result)
{
	result = float(1 - kSmoothDegree_) * prev_mask + float(kSmoothDegree_) * current_mask;
}


void VideoSmooth::SeqMaskFusion(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& fusion_result)
{
	int height = new_img.size().height;
	int width = new_img.size().width;
	// img & mask proprocessing
	cv::pyrDown(new_img, new_img);
	new_img.convertTo(new_img, CV_32FC3);
	img_seq_buffer_.push_back(new_img.clone()); // matÇ³¿½±´£¬ÐèÒªclone
	mask_seq_buffer_.push_back(new_mask.clone());
	// sum up mask
	if (mask_seq_sum_.empty()) mask_seq_sum_ = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
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
		cv::Mat fusion_mask;
		cv::transform(img_std, fusion_mask, cv::Matx13f(1, 1, 1));
		double min, max;
		cv::minMaxLoc(fusion_mask, &min, &max);
		int max_threshold = 50; //**TODO**: hardcode threhold, might lead to performance hit under different environment
		if (max < max_threshold)
		{
			mask_inter_.copyTo(fusion_result);
		}
		else
		{
			fusion_mask /= max;
			fusion_mask *= 255.;
			fusion_mask.convertTo(fusion_mask, CV_8UC1);
			/*cv::namedWindow("img_std");
			cv::imshow("img_std", fusion_mask);*/
			cv::threshold(fusion_mask, fusion_mask, 0, 1, cv::THRESH_BINARY+cv::THRESH_OTSU);
			dilate(fusion_mask, fusion_mask, cv::Mat(), cv::Point(-1, -1), 3); // 3times
			pyrUp(fusion_mask, fusion_mask);
			/*cv::namedWindow("img_std_binary");
			cv::imshow("img_std_binary", fusion_mask * 255);*/
			//fusion_result = fusion_mask * new_mask + (1 - fusion_mask) * mask_inter_;
			cv::copyTo(new_mask, fusion_result, fusion_mask); // directly copy pixel value with mask, which yields identic result as above
			cv::copyTo(mask_inter_, fusion_result, 1 - fusion_mask);
		}
		// ===========================fusion===============================
		// erase the oldest img/mask pair
		mask_seq_sum_ -= mask_seq_buffer_[0];
		img_seq_sum_ -= img_seq_buffer_[0];
		img_seq_buffer_.erase(img_seq_buffer_.begin());
		mask_seq_buffer_.erase(mask_seq_buffer_.begin());
	}
}


void VideoSmooth::Process(cv::Mat& new_img, cv::Mat& new_mask, cv::Mat& result)
{
	// memory binding
	SeqMaskFusion(new_img, new_mask, result);
	//cv::imshow("fusion", result * 255);
	result.convertTo(result, CV_32FC1); // convertion to fp32 for alpha-smooth later
	if (!(prev_mask_.empty()))
	{
		MaskSmooth(prev_mask_, result, result);
		EdgeSmooth(result, 1);
		result.copyTo(prev_mask_);
	}
	else
	{
		EdgeSmooth(new_mask, 1);
		new_mask.copyTo(prev_mask_);
	}
}