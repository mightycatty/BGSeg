#include "roi_generator.h"
#include "utils.h"
using namespace std;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

static float SimpleSmooth(const float x_t, const float s_t_0, const float kAlpha)
{
	float s_t = kAlpha * x_t + (1 - kAlpha) * s_t_0;
	return s_t;
}


ROIGenerator::ROIGenerator()
{

}


ROIGenerator::~ROIGenerator()
{
}


cv::Rect ROIGenerator::getBoundingRectFromMask(const cv::Mat& alpha_mask)
{
	//  bounding rect extraction from the contour with largest area in the bianrymask
	// ATTENTION:under the policy above, only single human instance is supported for segmentation
	std::vector<std::vector<cv::Point> > contours;
	cv::Rect ROIRect;
	cv::Mat binary_mask;
	cv::threshold(alpha_mask, binary_mask, 0.1, 1, cv::THRESH_BINARY);
	binary_mask.convertTo(binary_mask, CV_8UC1);
	findContours(binary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size()> 0)
	{
		std::sort(contours.begin(), contours.end(), ContoursSortFun);
		ROIRect = boundingRect(contours[0]);
	}
	else
	{
		ROIRect = cv::Rect(0, 0, or_shape_[0], or_shape_[1]);
	}
	return ROIRect;
}


void ROIGenerator::ROISmooth(cv::Rect rectCurrent)
{
	int x_0 = SimpleSmooth(rectCurrent.x, ROI_rect_.x, kAlpha_);
	int y_0 = SimpleSmooth(rectCurrent.y, ROI_rect_.y, kAlpha_);
	int x_1 = SimpleSmooth((rectCurrent.x + rectCurrent.width), (ROI_rect_.x + ROI_rect_.width), kAlpha_);
	int y_1 = SimpleSmooth((rectCurrent.y + rectCurrent.height), (ROI_rect_.y + ROI_rect_.height), kAlpha_);
	ROI_rect_.x = x_0;
	ROI_rect_.y = y_0;
	ROI_rect_.width = x_1 - x_0;
	ROI_rect_.height = y_1 - y_0;
}


void ROIGenerator::getROIImage(cv::Mat& src, cv::Mat& dst)
{
	static double duration_ms;
	duration_ms = std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - last_time_stamp_).count();
	last_time_stamp_ = std::chrono::high_resolution_clock::now();
	if (duration_ms > reset_time_gap_)
	{
		Reset();
	}
	if (first_frame_flag_)
	{
		auto orHeight = src.size().height;
		auto orWidth = src.size().width;
		or_shape_.push_back(orWidth);
		or_shape_.push_back(orHeight);
		dst = src.clone();
		//return;
	}
	else
	{
		/*assert(!ROI_rect_.empty());
		assert(src.size().height == or_shape_[1]);
		assert(src.size().width == or_shape_[0]);*/
		cv::Mat roi_img = src(ROI_rect_);
		roi_img.copyTo(dst);
	}
	return;
}

void ROIGenerator::restoreFromROI(cv::Mat& maskSrc, cv::Mat& maskDst)
{
	if (!first_frame_flag_)
	{
		assert(!ROI_rect_.empty());
		assert(!or_shape_.empty());
		maskDst = cv::Mat::zeros(cv::Size(or_shape_[0], or_shape_[1]), CV_32FC1);
		maskSrc.copyTo(maskDst(ROI_rect_));
		//int top = (int)ROI_rect_.y;
		/*int left = (int)ROI_rect_.x;
		int right = (int)(or_shape_[0] - ROI_rect_.x - ROI_rect_.width);
		int bottom = (int)(or_shape_[1] - ROI_rect_.y - ROI_rect_.height);
		copyMakeBorder(maskSrc, maskDst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0.));*/
	}
	else
	{
		maskDst = maskSrc;
	}
}


void ROIGenerator::Reset()
{
	first_frame_flag_ = true;
	or_shape_.clear();
	//TODO:reset cv::rect to get TRUE output for rect.empry()
}

void ROIGenerator::ROICheck()
{
	int rect_x = (int)ROI_rect_.x;
	int rect_y = (int)ROI_rect_.y;
	int rect_x_1 = ROI_rect_.width + rect_x;
	int rect_y_1 = ROI_rect_.height + rect_y;
	// expand roi
	rect_x = max(int(0), (int)rect_x -  2 * kUpdatePixelsMargine_);
	rect_y = max(int(0), (int)rect_y - kUpdatePixelsMargine_);
	rect_x_1 = min(int(or_shape_[0]), rect_x_1 +  2 * kUpdatePixelsMargine_);
	rect_y_1 = min(int(or_shape_[1]), rect_y_1 + kUpdatePixelsMargine_);
	// reject small roi
	if ((rect_x_1 - rect_x) < (kRoiLeastSizeFraction_ * or_shape_[0]))
	{
		rect_x = 0;
		rect_x_1 = or_shape_[0];
	}
	if ((rect_y_1 - rect_y) < (kRoiLeastSizeFraction_ * or_shape_[1]))
	{
		rect_y = 0;
		rect_y_1 = or_shape_[1];
	}
	ROI_rect_ = cv::Rect(rect_x, rect_y, (rect_x_1 - rect_x), (rect_y_1 - rect_y));
}

void ROIGenerator::Update(const cv::Mat& mask)
{
	auto rectCurrent = getBoundingRectFromMask(mask);
	if (first_frame_flag_) 
	{
		first_frame_flag_ = false;
		ROI_rect_ = rectCurrent;
	}
	else
	{
		ROISmooth(rectCurrent);
	}
	ROICheck();
}

void ROIGenerator::Update(const std::vector<cv::Point>& largest_cnt)
{
	static cv::Rect rect_current;
	if (largest_cnt.size() > 1)
	{
		rect_current = boundingRect(largest_cnt);
	}
	else
	{
		rect_current = cv::Rect(0, 0, or_shape_[0], or_shape_[1]);
	}
	if (first_frame_flag_)
	{
		first_frame_flag_ = false;
		ROI_rect_ = rect_current;
	}
	else
	{
		ROISmooth(rect_current);
	}
	ROICheck();
}

void ROIGenerator::Update(const std::vector<cv::Point>& largest_cnt, const cv::Mat& roi_img, const int inference_shape)
{
	static cv::Rect rect_current;
	if (largest_cnt.size() > 1)
	{
		rect_current = boundingRect(largest_cnt);
		rect_current.x = rect_current.x / (float)inference_shape * roi_img.size().width + ROI_rect_.x;
		rect_current.width = rect_current.width / (float)inference_shape * roi_img.size().width;
		rect_current.y = rect_current.y / (float)inference_shape * roi_img.size().height + ROI_rect_.y;
		rect_current.height = rect_current.height / (float)inference_shape * roi_img.size().height;
	}
	else
	{
		rect_current = cv::Rect(0, 0, or_shape_[0], or_shape_[1]);
	}
	if (first_frame_flag_)
	{
		first_frame_flag_ = false;
		ROI_rect_ = rect_current;
	}
	else
	{
		ROISmooth(rect_current);
	}
	ROICheck();
}