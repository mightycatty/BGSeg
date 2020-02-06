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
bool ContoursSortFun(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
std::vector<cv::String> getModelInFolder(const cv::String model_folder);
