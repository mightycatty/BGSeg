#include "seg_sdk.h"

SegSdk::SegSdk(int model_index, bool force_cpu_mode, std::string cpu_threads, std::string model_folder)
	: ie_(model_index, cpu_threads, force_cpu_mode, model_folder)
{
	// force compact mode if model index is 0 and none gpu is detected
	if ((model_index == 0) && force_cpu_mode)
	{
		compact_mode_ = true;
	}
	cv::setNumThreads(0);
}

SegSdk::~SegSdk()
{
	//
}

// unneccesary when embeded in OpenvinoIE
void preprocessing(cv::Mat& kInputImg, int resize_width, std::string cvt_color)
{
	//ResizeWithPadding(kInputImg, output_img, resize_width);
	if (cvt_color != "RGB")
	{
		cv::cvtColor(kInputImg, kInputImg, cv::COLOR_BGR2RGB);
	}
	if ((kInputImg.size().width != resize_width) || (kInputImg.size().height != resize_width))
	{
		cv::resize(kInputImg, kInputImg, cv::Size(resize_width, resize_width));
	}
	return;
}


// trim padding and restore to the original shape
void restoreShape(cv::Size or_shape, cv::Mat& mask)
{
	int or_height = or_shape.height;
	int or_width = or_shape.width;
	resize(mask, mask, cv::Size(or_width, or_height));
}


bool SegSdk::segImg(cv::Mat& inputImg, cv::Mat& segResult, std::string cvt_color)
{
	static cv::Mat img_roi, roi_seg_result;
	static cv::Size img_roi_or_size;
	// ============================================== preprocessing =================================
	roi_generator_.getROIImage(inputImg, img_roi);
	img_roi_or_size = img_roi.size();
	preprocessing(img_roi, ie_.input_shape_, cvt_color);
	// ============================================= model inference ================================
	if (!ie_.Predict(img_roi, roi_seg_result)) { return false; }
	// ============================================== postprocessing =================================
	cv::threshold(roi_seg_result, roi_seg_result, 0.5, 1, cv::THRESH_BINARY);
	restoreShape(img_roi_or_size, roi_seg_result);
	roi_generator_.restoreFromROI(roi_seg_result, segResult);
	post_processor_.Process(inputImg, segResult, segResult, compact_mode_); //TODO: BUGS when nothing detected
	roi_generator_.Update(post_processor_.relative_rect_);
	return true;
}


