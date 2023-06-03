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
#include <filesystem>
#include <stdlib.h>
#include <process.h>
#include <wchar.h>

using namespace std;
using namespace cv;


std::string GetCurrentDirectory()
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");

	return std::string(buffer).substr(0, pos);
}

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

void detectObjectAndDisplayInEllipse(Mat frame, CascadeClassifier obj_cascade)
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
		cout << "X: " << center.x << "; Y: " << center.y << "\n";
		ellipse(frame, center, Size(objs[i].width / 2, objs[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);


	}

	//-- Show what you got
	imshow("Capture - Face detection", frame);
	cv::waitKey(0);
}

void detectObjectAndDisplayInRectangle(Mat frame, CascadeClassifier obj_cascade)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> objs;
	obj_cascade.detectMultiScale(frame_gray, objs);
	for (size_t i = 0; i < objs.size(); i++)
	{
		cout << "X: " << objs[i].x << "; Y: " << objs[i].y << "\n";
		Rect r = Rect(objs[i].x, objs[i].y, objs[i].width, objs[i].height);

		cv::rectangle(frame, r, Scalar(255, 0, 255), 2, 8, 0);
		putText(frame, "Text in Images", Point(objs[i].x+10, objs[i].y+10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 255),2);//Putting the text in the matrix//
		//draw the rect defined by r with line thickness 1 and Blue color
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
	String objs_picture_lena_path = samples::findFile("data/lena.png");

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

	if (!objs_cascade.load(objs_cascade_name))
	{
		cout << "--(!)Error loading objs cascade\n";
		return -1;
	};




	// Dùng TinyXml
	//1. Load các dữ liệu cần thiết để quét vật thể: vector các tags(.xml), vector các vật thể(.xml), cascade của dữ liệu đánh  giá(.xml), ảnh test lena(.png)


	// Đọc cascade
	TiXmlDocument cascade_objs_doc("data/objs_cascade.xml");
	//doc.LoadFile("data/objs_cascade.xml");
	if (!cascade_objs_doc.LoadFile())
	{
		printf("%s", cascade_objs_doc.ErrorDesc());
		return -1;
	}
	else {

		TiXmlNode* opencv_storage = cascade_objs_doc.FirstChildElement();
		//Tìm phần tử con đầu tiên của node roor
		TiXmlNode* cascade = opencv_storage->FirstChildElement();
		//cout <<"cascade: "<< cascade->ToElement()/*Chuyển phần tử thành element */->Attribute("type_id")/*Lấy ra type_id của casecade*/ << endl;
	}


	//Đọc vector tags, vector vật thể
	vector<string> vec_tags;
	vector<string> vec_objs;
	TiXmlDocument classify_objs_doc("data/classify_objs.xml");
	//doc.LoadFile("data/objs_cascade.xml");
	if (!classify_objs_doc.LoadFile())
	{
		printf("%s", classify_objs_doc.ErrorDesc());
		return -1;
	}
	else {

		TiXmlNode* objects_info = classify_objs_doc.FirstChildElement();
		TiXmlNode* tag_list = objects_info->FirstChildElement();

		for (TiXmlNode* tags = tag_list->FirstChildElement(); tags != NULL; tags = tags->NextSibling()) {
			cout <<"Tag: " << tags->Value()<<"\n";
			vec_tags.push_back(tags->Value());
			for (TiXmlNode* objs = tags->FirstChildElement(); objs != NULL; objs = objs->NextSibling()) {
				cout << objs->ToElement()->GetText() << "\n";
				vec_objs.push_back(objs->ToElement()->GetText());

			}

		}
	}


	//Load ảnh test lena
	Mat img_lena = imread(objs_picture_lena_path, IMREAD_COLOR);
	if (img_lena.empty())
	{
		std::cout << "Could not read the image: " << objs_picture_lena_path << std::endl;
		return 1;
	}

	Mat img = imread(objs_picture_path, IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "Could not read the image: " << objs_picture_path << std::endl;
		return 1;
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

	//detectObjectAndDisplayInEllipse(img, face_cascade_name);
	//detectObjectAndDisplayInRectangle(img_lena, face_cascade_name);
	//detectObjectAndDisplayInRectangle(img, face_cascade_name);

	//int i;
	//printf("Checking if processor is available...");
	//if (system(NULL)) puts("Ok");
	//else exit(EXIT_FAILURE);
	//printf("Executing command DIR...\n");
	//i = system("dir");
	//printf("The value returned was: %d.\n", i);




	cout << "my directory is " << GetCurrentDirectory() << "\n";

	//cd to the folder first
	string cdConsoleCommand = "cd " + GetCurrentDirectory()+"\\ffmpeg";

	/*Nhắc nhở, chỉ đc sử dụng 1 dòng system("command") thôi
	//const char* commandcdConsoleCommand = cdConsoleCommand.c_str();
	//cout << "Compiling: "<<cdConsoleCommand << endl;
	//system(commandcdConsoleCommand);
	//cout << GetCurrentDirectory()<<endl;
	*/


	string filename="testVD1.mp4";
	std::size_t found = filename.find_last_of(".");
	string filenamewithoutext = filename.substr(0, found);
	string prepeareFolderForThumbnailsCommand = "prepareFolder " + filenamewithoutext;

	string thubnailshotsCommand = "ffmpeg -i "+filename+" -vf fps=1/20 "+ filenamewithoutext +"\\img%03d.png";
	string str = cdConsoleCommand +" && "+prepeareFolderForThumbnailsCommand + " && " + thubnailshotsCommand + " && ffplay " + filename;

	// Convert string to const char * as system requires
	// parameter of type const char *
	const char* command = str.c_str();

	cout << "Compiling file using " << command << endl;
	system(command);

	cout << "\nRunning file ";

	getchar();

	return 0;
}