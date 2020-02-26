#ifndef __CONFIG_H_
#define __CONFIG_H_
#include <string>
#include <opencv2/opencv.hpp>
struct SegSdkConfig
{
	//--------------------------------device config----------------------------------------
	std::string device = "CPU"; //""
	std::string cpu_thread = "1"; // only apply on CPU mode
	//--------------------------------network config----------------------------------------
	const std::string model_folder = "../model_IR/256_256/fp32/";
	const std::string cpu_extention = "cpu_extension_avx2.dll";
	const std::string model_name = "mobilenet_3.5B_with_mean-scale";
	const std::string model_xml = model_folder + model_name + ".xml";
	const std::string bin_file_name = model_folder + model_name + ".bin";
	const int width = 256;
	const int height = width;
	const cv::Size desize = cv::Size(width, height);
	const std::string plugin_dirs = "";
};
#endif