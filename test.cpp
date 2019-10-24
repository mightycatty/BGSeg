// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "seg_sdk.h"


void ImreadBackground(cv::Mat& bg_mat)
{
	std::string bg_dir = "../samples/bg.png";
	bg_mat = cv::imread(bg_dir);
}

void testWithWebCam()
{
	SegSdk segSdk;
	cv::Mat segResult;
	bool staticFlag = false;
	// resource initialization
	cv::VideoCapture cap(0);
	if (cap.isOpened())
	{
		cv::Mat frame;
		while (1)
		{
			cap >> frame;
			
			segSdk.segImg(frame, segResult, staticFlag);
			/*cv::namedWindow("Cam", cv::WINDOW_NORMAL);
			cv::imshow("Cam", frame);*/
			////cv::waitKey(0);
			//cv::namedWindow("Mask", cv::WINDOW_NORMAL);
			//cv::imshow("Mask", segResult*255);
			cv::waitKey(1);
		}
		
	}
}



int main(int argc, char *argv[]) 
{
	try
	{
		testWithWebCam();
	}
	catch (const std::exception &exc)
	{
		std::cerr << exc.what();
		auto a = 1;
	}
}
	