#pragma 
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <inference_engine.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <opencv2/opencv.hpp>


std::vector<cv::String> getImageDirInFolder(cv::String pattern);
InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat);
bool ContoursSortFun(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
void ResizeWithPadding(const cv::Mat& img, cv::Mat& result, int target_width);

