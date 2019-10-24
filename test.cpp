// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "seg_sdk.h"


void ImreadBackground(cv::Mat& bg_mat)
{
	std::string bg_dir = "../samples/bg.png";
	bg_mat = cv::imread(bg_dir);
}

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
void testWithWebCam()
{
	SegSdk segSdk;
	cv::Mat segResult;
	bool staticFlag = false;
	// resource initialization
	cv::VideoCapture cap(0);
	if (cap.isOpened())
	{
		//bool bTemp = true;
		//cv::Mat frame = cv::imread("D:/Work/Test/seg_demo_openvino19/samples/bg.png");
		cv::Mat frame;

		while (1)
		{
			cap >> frame;

			auto t0 = std::chrono::high_resolution_clock::now();
			if (segSdk.segImg(frame, segResult, staticFlag)) {
				/*cv::namedWindow("Cam", cv::WINDOW_NORMAL);
				cv::imshow("Cam", frame);
				cv::namedWindow("Mask", cv::WINDOW_NORMAL);
				cv::imshow("Mask", segResult * 255);
				cv::waitKey(1);*/
			}
			auto t1 = std::chrono::high_resolution_clock::now();
			double duration_ms = std::chrono::duration_cast<ms>(t1 - t0).count();
			//std::cout << duration_ms << " milliseconds" << std::endl;

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
	