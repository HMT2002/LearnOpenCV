#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "tinyxml.h"


#include <windows.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>


using namespace std;
using namespace cv;


void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier objs_cascade;


void TakePicture() {
	cv::Mat img = cv::imread("C:/Users/blues/Desktop/image.jpg");
	namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
	cv::imshow("First OpenCV Application", img);
	cv::moveWindow("First OpenCV Application", 0, 45);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		Mat faceROI = frame_gray(faces[i]);
		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}

void detectObjectAndDisplay(Mat frame, CascadeClassifier obj_cascade)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> objs;
	obj_cascade.detectMultiScale(frame_gray, objs);
	for (size_t i = 0; i < objs.size(); i++)
	{
		Point center(objs[i].x + objs[i].width / 2, objs[i].y + objs[i].height / 2);
		ellipse(frame, center, Size(objs[i].width / 2, objs[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
	}

	//-- Show what you got
	imshow("Capture - Face detection", frame);
	cv::waitKey(0);

}

int main(int argc, const char** argv)
{


	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{objs_cascade|data/objs_cascade.xml|Path to objs cascade.}"

		"{camera|0|Camera device number.}");

	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();


#pragma region Load dữ liệu lên

	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
	String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
	String objs_cascade_name = samples::findFile(parser.get<String>("objs_cascade"));
	String objs_picture_path = samples::findFile("data/image.jpg");
	cout << objs_picture_path + "\n";
	cout << objs_cascade_name + "\n";


	//-- 1. Load the cascades
	//if (!face_cascade.load(face_cascade_name))
	//{
	//    cout << "--(!)Error loading face cascade\n";
	//    return -1;
	//};
	//if (!eyes_cascade.load(eyes_cascade_name))
	//{
	//    cout << "--(!)Error loading eyes cascade\n";
	//    return -1;
	//};


	Mat img = imread(objs_picture_path, IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "Could not read the image: " << objs_cascade_name << std::endl;
		return 1;
	}
	if (!objs_cascade.load(objs_cascade_name))
	{
		cout << "--(!)Error loading objs cascade\n";
		return -1;
	};

	TiXmlDocument doc("data/objs_cascade.xml");
	//doc.LoadFile("data/objs_cascade.xml");
	if (!doc.LoadFile())
	{
		printf("%s", doc.ErrorDesc());
		return -1;
	}
	else {

		TiXmlNode* opencv_storage = doc.FirstChildElement();
		//Tìm phần tử con đầu tiên của node roor
		TiXmlNode* cascade = opencv_storage->FirstChildElement();
		//cout <<"cascade: "<< cascade->ToElement()/*Chuyển phần tử thành element */->Attribute("type_id")/*Lấy ra type_id của casecade*/ << endl;
	}
	string line;
	ifstream infile("data/classify_objs.txt");
	vector<string> vec_objs;
	while (getline(infile, line))
	{
		std::istringstream iss(line);
		//cout << line << "\n";
		vec_objs.push_back(line);
		cout << "vector: " << line << "\n";
	}
	if (vec_objs.size() == 0) {
		cout << "No line in file or file not exist! \n";
	}
	else {
		cout<<"There are " << vec_objs.size()<<" objects to identify! \n";
	}

#pragma endregion

	//int camera_device = parser.get<int>("camera");
	//VideoCapture capture;
	////-- 2. Read the video stream
	//capture.open(camera_device);
	//if (!capture.isOpened())
	//{
	//    cout << "--(!)Error opening video capture\n";
	//    return -1;
	//}

	//Mat frame;
	//while (capture.read(frame))
	//{
	//    if (frame.empty())
	//    {
	//        cout << "--(!) No captured frame -- Break!\n";
	//        break;
	//    }
	//    //-- 3. Apply the classifier to the frame
	//    //detectAndDisplay(frame);
	//    detectObjectAndDisplay(frame, objs_cascade);
	//    if (waitKey(10) == 27)
	//    {
	//        break; // escape
	//    }
	//}



	//detectObjectAndDisplay(img, objs_cascade_name);

	getchar();

	return 0;
}