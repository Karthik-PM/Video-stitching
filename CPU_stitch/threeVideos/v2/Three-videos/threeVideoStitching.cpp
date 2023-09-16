#include<bits/stdc++.h>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/features2d.hpp>

int main(int argc, char const *argv[])
{
    cv::VideoCapture video1("videos/output2.mkv");
    while(video1.isOpened()){
        cv::Mat Frame1;
        cv::Mat Frame2;
        cv::Mat Frame3;
        video1.read(Frame1);
        cv::imshow("Video1", Frame1);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }

    video1.release();
    cv::destroyAllWindows();
    return 0;
}
