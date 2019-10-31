#include "utils.h"


std::vector<cv::String> getImageDirInFolder(cv::String pattern)
{
	std::vector<cv::String> imageDirList;
	cv::glob(pattern, imageDirList, false);
	return imageDirList;
}

bool ContoursSortFun(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

void ResizeWithPadding(const cv::Mat& img, cv::Mat& square, int target_width = 256)
{
	int width = img.cols,
		height = img.rows;

	square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return;
}
