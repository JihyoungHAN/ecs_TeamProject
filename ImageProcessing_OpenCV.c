//#include "vision.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
#define PI 3.141592
//Setting Variable
#define POSITION_Y 151 //will be adjuted
#define FRAME_WIDTH 320
//HSV Color Condition (YELLOW)
#define Y_H_LOW 20
#define Y_H_HIGH 70
#define Y_S_LOW 60
#define Y_S_HIGH 250
#define Y_V_LOW 30
#define Y_V_HIGH 255
//HSV Color Condition (Red)
#define R_H_LOW 0
#define R_H_HIGH 60
#define R_S_LOW 10
#define R_S_HIGH 255
#define R_V_LOW 10
#define R_V_HIGH 255
//HSV Color Condition (Green)
#define G_H_LOW 60
#define G_H_HIGH 160
#define G_S_LOW 0
#define G_S_HIGH 255
#define G_V_LOW 0
#define G_V_HIGH 255
//HSV Color Condition (White)
#define W_H_LOW 0
#define W_H_HIGH 360
#define W_S_LOW 0
#define W_S_HIGH 45
#define W_V_LOW 230
#define W_V_HIGH 255
 
vector<Point> FindLocalMaxima(Mat &src); //Funtion we use to find cornerpoints

int main()
{
    /*------------------------영상전처리----------------------*/
    //Mat Img = imread("/home/yeogsam/opencv/trafic_signal/imgOrigin0.png",1);//Red 컬러로 
    //Mat Img = imread("/home/yeogsam/opencv/trafic_signal/imgOrigin8.png",1);//Yellow 컬러로
    //Mat Img = imread("/home/yeogsam/opencv/trafic_signal/imgOrigin10.png",1);//Left 컬러로
    Mat Img = imread("/home/yeogsam/opencv/trafic_signal/imgOrigin16.png",1);//Right 컬러로
    //Mat inputImage(inputImage);
    cout<<"Rows: "<<Img.rows<<" Columns: "<<Img.cols<<endl;
    cout<<"Dims: "<<Img.dims<<" Total: "<<Img.total()<<endl;
    cout<<"Depth: "<<Img.depth()<<" Channel: "<<Img.channels()<<endl;
    cout<<"Type: "<<Img.type()<<" Size: "<<Img.size()<<endl;
     
    Mat cornerMap, Green_Binary_Img;
    Mat Hsv_Img, Gray_Img, FilteredImage; 
    cvtColor(Img, Hsv_Img, COLOR_BGR2HSV);
    cvtColor(Img, Gray_Img, CV_BGR2GRAY );
    /*영상 블러 , 에지 부각 */
    /*int border = 3; // 7x7
    Size ksize(border * 2 + 1, border * 2 + 1);
    int d = ksize.width;*/
    double sigmaColor = 10.0;
    double sigmaSpace = 10.0;
      
    bilateralFilter(Gray_Img, FilteredImage, -1, sigmaColor, sigmaSpace);
    //imshow("dstImage", FilteredImage);
    /*------------------------------------------------------*/
    /*--------------원 검출-----------------*/
    vector<Vec3f> circles;
    HoughCircles(FilteredImage, circles, CV_HOUGH_GRADIENT, 2, 30, 200, 50, 10, 50); //원 그리는 함수 
    //image,circles,method,dp,minDist,param1,param2,minRadius,maxRadius
    for(int i=0; i<circles.size(); i++){cout<<"center: "<<circles[i][0]<<endl;}//각 원 순서대로 중심점 색 정보 출력
    /*-------------------------------------*/
    /*-------------검출 된 원 내부 색상정보----------------*/
    for(size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]),cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        //circle(Gray_Img, center, 3, Scalar(0,0,255), -1, 8, 0);//circle center 화면에 원 그리는 함수 
        //circle(Gray_Img, center, radius, Scalar(0,0,255), 3, 8, 0);//outline  화면에 원 그리는 함수 
        cout<<"center: "<<center<<"radius: "<<endl;
        //imshow("Gray_Img",Gray_Img);
         
        int k=circles[i][0];//x 
        int j=circles[i][1];//y
        //--------------
        cout<<"k: "<<k<<"j: "<<j<<endl;
        int B=Img.at<Vec3b>(j,k)[0]; //화소접근시 x,y좌표 반대로 되므로 주의!!! 
        int G=Img.at<Vec3b>(j,k)[1];
        int R=Img.at<Vec3b>(j,k)[2];
         
        cout<<"B"<<B<<endl;
        cout<<"G"<<G<<endl;
        cout<<"R"<<R<<endl;
         
        if(R-B>20 && R-G>20 && R>100){cout<<"Stop due to the RedLight"<<endl;} 
         
        if(G-R>20 && G-B>20 && G>100)
        {
            cout<<"Detected GreenLight"<<endl;
            int CenterX = circles[i][0];
            int CenterY = circles[i][1];
            int Cradius = circles[i][2]; //반지름 
            int Width = 2*circles[i][2]; //width=height (circle) -> rect로 근사 
            int Height = 2*circles[i][2];
            int Green_cnt,Green_data,Red_data,Blue_data;    
            int TotalPixel = cvFloor(PI*Cradius*Cradius);
             
            for(int k=CenterX-Width; k<=CenterX+Width; k++)//This needs to be made
            {
                for(int j=CenterY-Height; j<=CenterY+Height; j++)
                {
                    int Blue_data = Img.at<Vec3b>(j,k)[0];
                    int Green_data = Img.at<Vec3b>(j,k)[1];
                    int Red_data = Img.at<Vec3b>(j,k)[2];
                     
                    if(Green_data-Blue_data>20 && Green_data-Red_data>20 && Green_data>100){
                        Green_cnt++; 
                    }
                }
            }cout<<"Green_cnt: "<<Green_cnt<<" TotalPixel: "<<TotalPixel<<endl;
             
            if(TotalPixel*0.6<Green_cnt){cout<<"--------------------------GoRight"<<endl;}
            else{cout<<"---------------------------GoLeft"<<endl;}
        }      
         
        else{cout<<"stay stop!"<<endl;}
    }
    //imshow("Green_ROI", Green_Binary_Img);
    //imshow("Origin image",Img);
    //waitKey(0);
    return 0;
}