#include "utils.h"


InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat) {
	size_t channels = mat.channels();
	size_t height = mat.size().height;
	size_t width = mat.size().width;

	size_t strideH = mat.step.buf[0];
	size_t strideW = mat.step.buf[1];

	bool is_dense =
		strideW == channels &&
		strideH == channels * width;

	if (!is_dense) THROW_IE_EXCEPTION
		<< "Doesn't support conversion from not dense cv::Mat";

	InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
		{ 1, channels, height, width },
		InferenceEngine::Layout::NHWC);

	return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

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
