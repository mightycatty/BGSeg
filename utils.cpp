#include "utils.h"


std::vector<cv::String> getImageDirInFolder(cv::String pattern)
{
	std::vector<cv::String> imageDirList;
	cv::glob(pattern, imageDirList, false);
	return imageDirList;
}

std::vector<cv::String> getModelInFolder(const cv::String model_folder)
{
	std::vector<cv::String> xml_dir, bin_dir;
	cv::glob(model_folder + "/*.xml", xml_dir, false);
	cv::glob(model_folder + "/*.bin", bin_dir, false);
	xml_dir.insert(xml_dir.end(), bin_dir.begin(), bin_dir.end());
	return xml_dir;
}

bool ContoursSortFun(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}