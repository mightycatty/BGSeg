// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "seg_sdk.h"
#include <chrono>
#include <thread>


static void game_blending(cv::Mat& fg, cv::Mat& bg, cv::Mat& alpha)
{
	vector<cv::Mat> channels, beta_v;
	cv::Mat beta = 1. - alpha;
	channels.push_back(alpha);
	channels.push_back(alpha);
	channels.push_back(alpha);
	cv::merge(channels, alpha);
	beta_v.push_back(beta);
	beta_v.push_back(beta);
	beta_v.push_back(beta);

	cv::merge(beta_v, beta);

	cv::Rect blending_roi(1200, 550, fg.size().width, fg.size().height);

	fg.convertTo(fg, CV_32FC3);
	cv::multiply(fg, alpha, fg);
	fg.convertTo(fg, CV_8UC3);

	cv::Mat bg_roi = bg(blending_roi).clone();
	//cv::imshow("blending_path", alpha);

	bg_roi.convertTo(bg_roi, CV_32FC3);

	cv::multiply(bg_roi, beta, bg_roi);
	bg_roi.convertTo(bg_roi, CV_8UC3);
	cv::Mat blending_patch = fg + bg_roi;
	blending_patch.copyTo(bg(blending_roi));
}

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
void testWithWebCam()
{
	SegSdk segSdk("CPU");
	cv::Mat segResult;
	bool staticFlag = false;
	// resource initialization
	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FPS, 30);
	//cv::VideoCapture cap_bg("../samples/bg.mp4");
	if (cap.isOpened())
	{
		//bool bTemp = true;
		cv::Mat frame, bg, blend_img;
		for (int i = 0 ; i < 100; ++i)
		{
			cap >> frame;
		}
		cap.release();
		while (1)
		{
			/*cap >> frame;
			try
			{
				cap_bg >> bg;
				cap_bg >> bg;
			}
			catch (const std::exception &exc)
			{
				cv::VideoCapture cap_bg("../samples/bg.mp4");
				cap_bg >> bg;
			}*/
			auto t0 = std::chrono::high_resolution_clock::now();
			if (segSdk.segImg(frame, segResult)) {
				
				/*cv::namedWindow("Cam", cv::WINDOW_NORMAL);
				cv::imshow("Cam", frame);
				cv::namedWindow("Mask", cv::WINDOW_NORMAL);
				cv::Mat mask_vis;
				mask_vis = segResult * 255;
				mask_vis.convertTo(mask_vis, CV_8UC1);
				cv::imshow("Mask", mask_vis);
				cv::waitKey(1);*/
				// ================ blending ==========================
				/*game_blending(frame, bg, segResult);
				cv::namedWindow("blending", cv::WINDOW_NORMAL);
				cv::imshow("blending", bg);
				cv::waitKey(1);*/
			}
			auto t1 = std::chrono::high_resolution_clock::now();
			double duration_ms = std::chrono::duration_cast<ms>(t1 - t0).count();
			float fps_control = 30.;
			float f_duration = 1000. / fps_control;
			if (duration_ms < f_duration)
			{
				long sleep_time = f_duration - duration_ms;
				std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
			}
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
	