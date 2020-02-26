#include "roi_generator.h"
#include "utils.h"
using namespace std;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

static float SimpleSmooth(const float x_t, const float s_t_0, const int margine, float alpha)
{
	if (abs(x_t - s_t_0) >= margine)
	{
		alpha = 0.5;
	}
	float s_t = alpha * x_t + (1 - alpha) * s_t_0;
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
	cv::threshold(alpha_mask, binary_mask, 0.5, 1, cv::THRESH_BINARY);
	binary_mask.convertTo(binary_mask, CV_8UC1);
	findContours(binary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() > 0)
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
	static int update_pixel_witdh, update_pixel_height;
	update_pixel_witdh = int(or_shape_[0] * kUpdatePixelsMargineFraction_);
	update_pixel_height = int(or_shape_[1] * kUpdatePixelsMargineFraction_);
	int x_0 = SimpleSmooth(rectCurrent.x, ROI_rect_.x, update_pixel_witdh, kUpdateFactor_);
	int y_0 = SimpleSmooth(rectCurrent.y, ROI_rect_.y, update_pixel_height, kUpdateFactor_);
	int x_1 = SimpleSmooth((rectCurrent.x + rectCurrent.width), (ROI_rect_.x + ROI_rect_.width), update_pixel_witdh, kUpdateFactor_);
	int y_1 = SimpleSmooth((rectCurrent.y + rectCurrent.height), (ROI_rect_.y + ROI_rect_.height), update_pixel_height, kUpdateFactor_);
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
	if (duration_ms > reset_time_gap_) Reset();
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
		src(ROI_rect_).copyTo(dst);
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
	static int update_pixel_witdh, update_pixel_height;
	update_pixel_witdh = int(or_shape_[0] * kUpdatePixelsMargineFraction_);
	update_pixel_height = int(or_shape_[1] * kUpdatePixelsMargineFraction_);
	rect_x = max(int(0), (int)rect_x - update_pixel_witdh);
	rect_y = max(int(0), (int)rect_y -update_pixel_height);
	rect_x_1 = min(int(or_shape_[0]), rect_x_1 + update_pixel_witdh);
	rect_y_1 = min(int(or_shape_[1]), rect_y_1 + update_pixel_height);
	// reject small roi
	static float pixel_fraction_threshold = 0.2;
	if ((rect_x_1 - rect_x) < (pixel_fraction_threshold * or_shape_[0]))
	{
		rect_x = 0;
		rect_x_1 = or_shape_[0];
	}
	if ((rect_y_1 - rect_y) < (pixel_fraction_threshold * or_shape_[1]))
	{
		rect_y = 0;
		rect_y_1 = or_shape_[1];
	}
	ROI_rect_ = cv::Rect(rect_x, rect_y, (rect_x_1 - rect_x), (rect_y_1 - rect_y));
}

void ROIGenerator::Update(const std::vector<float>& relative_rect)
{
	static cv::Rect rect_current;
	rect_current.x = int(relative_rect[0] * or_shape_[0]);
	rect_current.width = int(relative_rect[2] * or_shape_[0]);
	rect_current.y = int(relative_rect[1] * or_shape_[1]);
	rect_current.height = int(relative_rect[3] * or_shape_[1]);

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

void ROIGenerator::Update(cv::Mat& mask_result)
{
	static cv::Rect rect_current;
	rect_current = getBoundingRectFromMask(mask_result);
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
