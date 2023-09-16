#include<bits/stdc++.h>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/features2d.hpp>

int main(int argc, char const *argv[])
{
    cv::VideoCapture video1("videos/output6.mkv");
    cv::VideoCapture video2("videos/output2.mkv");
    cv::VideoCapture video3("videos/output4.mkv");

    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::Mat Frame3;
    while(video1.isOpened() && video2.isOpened() && video3.isOpened()){
        bool isFrame1Active = video1.read(Frame1);
        bool isFrame2Active = video2.read(Frame2);
        bool isFrame3Active = video3.read(Frame3);

        // loop back video
        if(!isFrame1Active){
            video1.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        if(!isFrame2Active){
            video2.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        if(!isFrame3Active){
            video3.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // displaying the videos
        cv::imshow("Video1", Frame1);
        cv::imshow("Video2", Frame2);
        cv::imshow("Video3", Frame3);

        int key = cv::waitKey(100);
        if (key == 'q' || key == 27) {  // 'q' key or Esc key (27) to exit
            break;
        }
    }

    video1.release();
    video2.release();
    video3.release();
    cv::destroyAllWindows();
    return 0;
}
