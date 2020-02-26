// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "seg_sdk.h"
#include <chrono>
#include <thread>


static void blending(cv::Mat& fg, cv::Mat& bg, cv::Mat& alpha, int roi_fraction=2)
{
	cv::resize(fg, fg, cv::Size(bg.size().width / roi_fraction, bg.size().height / roi_fraction));
	cv::resize(alpha, alpha, cv::Size(bg.size().width / roi_fraction, bg.size().height / roi_fraction));
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

	cv::Rect blending_roi(bg.size().width / roi_fraction * (roi_fraction - 1), bg.size().height / roi_fraction * (roi_fraction - 1), fg.size().width, fg.size().height);

	fg.convertTo(fg, CV_32FC3);
	cv::multiply(fg, alpha, fg);
	fg.convertTo(fg, CV_8UC3);

	cv::Mat bg_roi = bg(blending_roi).clone();
	//cv::imshow("blending_path", alpha);

	bg_roi.convertTo(bg_roi, CV_32FC3);

	cv::multiply(bg_roi, beta, bg_roi);
	bg_roi.convertTo(bg_roi, CV_8UC3);
	bg_roi = fg + bg_roi;
	bg_roi.copyTo(bg(blending_roi));
}

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

bool PROFILE_FLAG = false;

void testWithWebCam()
{
	SegSdk segSdk(2, true, "1", "../models/");
	cv::Mat segResult;
	bool staticFlag = false;
	// resource initialization
	//cv::VideoCapture cap("F:\\heshuai\\data\\segmentation\\legacy\\videos\\×ø×Ë\\3680c56f859be76bb5ab3893eceab6b0.mp4");
	cv::VideoCapture cap("D:\\WIN_20200119_17_21_17_Pro.mp4");

	cap.set(cv::CAP_PROP_FPS, 30);
	cv::VideoCapture cap_bg("F:/heshuai/proj\\people_seg_stack\\seg_demo_openvino19\\samples\\bg.mp4");
	cap_bg.set(cv::CAP_PROP_FPS, 30);
	if (cap.isOpened())
	{
		//bool bTemp = true;
		cv::Mat frame, frame_resized, blend_img, bg;
		//auto bg = cv::imread("F:/heshuai/proj/people_seg_stack/seg_demo_openvino19/samples/star.png");
	/*	for (int i = 0 ; i < 500; ++i)
		{
			cap >> frame;
		}*/
		if (PROFILE_FLAG)
		{
			cap >> frame;
			cv::resize(frame, frame_resized, cv::Size(480, 360));
			cap.release();
		}
		while (1)
		{
			if (!PROFILE_FLAG)
			{
				cap >> frame;
				cv::resize(frame, frame_resized, cv::Size(480, 360));
				bg = cv::imread("F:/heshuai/proj/people_seg_stack/seg_demo_openvino19/samples/star.png");
			}

			//cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
			/*try
			{*/
			//cap_bg >> bg;
			/*bg = cv::imread("F:/heshuai/proj/people_seg_stack/seg_demo_openvino19/samples/star.png");

			cv::resize(bg, bg, cv::Size(1280, 720));*/

			//cap_bg >> bg;
			/*}
			catch (const std::exception &exc)
			{
				cap_bg.release();
				cv::VideoCapture cap_bg("F:\heshuai\proj\people_seg_stack\seg_demo_openvino19\samples\bg.mp4");
				cap_bg >> bg;
			}*/
			auto t0 = std::chrono::high_resolution_clock::now();
			if (segSdk.segImg(frame_resized, segResult, "BGR")) {

				auto t1 = std::chrono::high_resolution_clock::now();
				double duration_ms = std::chrono::duration_cast<ms>(t1 - t0).count();
				float fps_control = 30.;
				float f_duration = 1000. / fps_control;
				if (duration_ms < f_duration)
				{
					long sleep_time = f_duration - duration_ms;
					std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
				}

				if (!PROFILE_FLAG)
				{
					std::cout << duration_ms << std::endl;

					cv:; resize(segResult, segResult, frame.size());
					cv::namedWindow("Cam", cv::WINDOW_NORMAL);
					cv::imshow("Cam", frame);
					cv::namedWindow("Mask", cv::WINDOW_NORMAL);
					cv::Mat mask_vis;
					mask_vis = segResult * 255;
					mask_vis.convertTo(mask_vis, CV_8UC1);
					cv::imshow("Mask", mask_vis);
					// ================ blending ==========================
					blending(frame, bg, segResult, 1);
					cv::namedWindow("blending", cv::WINDOW_NORMAL);
					cv::imshow("blending", bg);
					cv::waitKey(1);
				}
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
	