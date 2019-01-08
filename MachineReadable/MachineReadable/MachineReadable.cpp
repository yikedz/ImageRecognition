// MachineReadable.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// 简书笔记：https://www.jianshu.com/p/f8d412ab4ead
//

#include "pch.h"
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;

Mat loadImage();								// 加载图片并二值化
Mat gaintComponent(Mat src);					// 提取最大连通域
int numMatch(Mat src, int method);				// 数字模板匹配
void choiceRec(Mat src);						// 选择题部分分割与识别
void numberRec(Mat src);						// 数字信息分割与识别

Mat loadImage()
{
	Mat src = imread("001.jpg");
	if (src.empty())
	{
		cout << "IMAGE LOAD FAILED!" << endl;
		exit(0);
	}
	Mat gray, binary;
	cvtColor(src, gray, CV_BGR2GRAY);
	adaptiveThreshold(gray, binary, 255, 0, 1, 101, 10);
	return binary;
}

Mat gaintComponent(Mat src)
{
	// 查找连通域
	vector<vector<Point>>contours;
	findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// 最大连通域
	vector<Point>maxContour;
	double maxArea = 0.0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxContour = contours[i];
		}
	}
	// 转换为矩形框(boundingbox)
	Rect maxRect = boundingRect(maxContour);
	Mat result = src(maxRect);
	return result;
}

void choiceRec(Mat src)
{
	// 定位到选择题部分
	Mat roiBox = gaintComponent(src);
	Mat choiceBox = gaintComponent(~roiBox);
	choiceBox = ~choiceBox;
	// 选择题部分边界处理
	choiceBox = choiceBox(Rect(0, 15, choiceBox.cols, choiceBox.rows / 2 - 22));
	// 定位涂抹区间连通域
	vector<vector<cv::Point>> contours, answer;
	cv::findContours(choiceBox, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > 20 && area < 60)
			answer.push_back(contours[i]);
	}
	// 存储涂抹中心点
	vector<Moments> mu(answer.size());
	for (int i = 0; i < answer.size(); i++)
		mu[i] = moments(answer[i], false);
	// 计算中心矩:
	vector<Point2f> mc(answer.size());
	for (int i = 0; i < answer.size(); i++)
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	// 得到答案(使用选择题模板，涂抹部分的答案用小写字母标识)
	char answers[10][20] = {
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' },
	{ ' ','A','B','C','D',' ','A','B','C','D', ' ','A','B','C','D', ' ','A','B','C','D' } };
	int x = choiceBox.cols / 20;
	int y = choiceBox.rows / 10;
	for (size_t i = 0; i < mc.size(); i++)
	{
		int x_index = mc[i].x / x;
		int y_index = mc[i].y / y;
		answers[y_index][x_index] += 32;
	}
	cout << "choice:" << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			if (answers[i][j] >= 'a'&&answers[i][j] <= 'd')
				cout << answers[i][j] << " ";
		}
		cout << endl;
	}
}

// method: 
// CV_TM_SQDIFF		=0,
// CV_TM_SQDIFF_NORMED =1,
// CV_TM_CCORR         =2,
// CV_TM_CCORR_NORMED  =3,
// CV_TM_CCOEFF        =4,
// CV_TM_CCOEFF_NORMED =5
int numMatch(Mat src, int method)
{
	// 模板简单处理
	Mat model = imread("model.jpg", CV_BGR2GRAY);
	resize(model, model, Size(src.cols * 10, src.rows));// 调整模板至便于匹配的大小
	// 取数字部分连通域，排除干扰因素
	src = gaintComponent(~src);
	src = ~src;
	// 单独识别1，因为1的连通域最小，长宽比大
	if (src.rows / src.cols > 2)
		return 1;
	// 带识别图像处理(单通道变三通道)
	cv::Mat three_ch = Mat::zeros(src.rows, src.cols, CV_8UC3);
	vector<Mat>channels;
	for (int i = 0; i < 3; i++)
		channels.push_back(src);
	merge(channels, three_ch);
	// 用matchTemplate()函数匹配数字
	Mat result;
	matchTemplate(model, three_ch, result, method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	// 通过匹配的区域的中心点坐标定位数字
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	int ans = (minLoc.x + three_ch.cols / 2) / (model.cols / 10);
	return ans;
}

void numberRec(Mat src)
{
	// 定位选择题上方数字信息
	Mat roiImg = gaintComponent(src);
	Mat infoBox = roiImg(Rect(0, 10, roiImg.cols, roiImg.rows / 4 - 10));
	vector<vector<cv::Point>> contours, numBox;
	cv::findContours(infoBox, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > 100)
			numBox.push_back(contours[i]);
	}
	vector<Mat>num;
	for (size_t i = 0; i < numBox.size(); i++)
	{
		cv::Rect rect = cv::boundingRect(numBox[i]);
		cv::Mat result = infoBox(rect);
		num.push_back(result);
	}
	// num[0]:15764713111
	// num[1]:073008
	vector<int> phoneNum;
	vector<int> paperNum;
	Mat model = imread("model.jpg", CV_BGR2GRAY);
	for (int j = 0; j < 2; j++)
	{
		vector<vector<Point>> temp;
		num[j] = ~num[j];
		cv::findContours(num[j], temp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (size_t i = 0; i < temp.size(); i++)
		{
			double area = cv::contourArea(temp[i]);
			if (area > 200)
			{
				cv::Rect rect = cv::boundingRect(temp[i]);
				cv::Mat src = num[j](rect);
				// method=1：归一化的平方差(CV_TM_SQDIFF_NORMED)
				int ans = numMatch(src, 1);
				if (j == 0)
					phoneNum.push_back(ans);
				else
					paperNum.push_back(ans);
			}
		}
	}
	cout << "PhoneNumber:";
	for (int i = 10; i >= 0; i--)
		cout << phoneNum[i];
	cout << endl << "PaperNumber:";
	for (int i = 5; i >= 0; i--)
		cout << paperNum[i];
	cout << endl;
}

int main()
{
	Mat src = loadImage();
	numberRec(src);
	choiceRec(src);
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
