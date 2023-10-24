#include<bits/stdc++.h>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/features2d.hpp>

//globlal declaration
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

std::vector<cv::KeyPoint> genrateKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint> keypoints){
    sift->detect(frame, keypoints);
    return keypoints;
}

cv::Mat genrateDescriptors(cv::Mat frame, std::vector<cv::KeyPoint> keypoints){
    cv::Mat descriptor;
    sift->compute(frame, keypoints, descriptor);
    return descriptor;
}

cv::Mat shiftImage(cv::Mat image, int x , int y){
    cv::Mat shift = (
        cv::Mat_<double>(3, 3) << 
        1, 0, x,
        0, 1, y,
        0, 0, 1
    );
    cv::warpPerspective(image, image, shift, image.size() * 2);
    return image;
}

std::vector<cv::DMatch> LowesRatioClean(std::vector<std::vector<cv::DMatch>> rawmatches){
    std::vector<cv::DMatch> goodMatches;

    
    double ratio = 0.8;
    for(auto match : rawmatches){
        // std::cout << match[0].distance << " " << match[1].distance << "\n";
        if(match[0].distance < ratio * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    return goodMatches;
}

int main(int argc, char const *argv[])
{
    cv::VideoCapture video1("videos/video_sample2/left.mp4");
    cv::VideoCapture video2("videos/video_sample2/middle.mp4");
    cv::VideoCapture video3("videos/video_sample2/right.mp4");

    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::Mat Frame3;

    cv::Mat Tranformation21;
    cv::Mat Tranformation23;

    bool isHomographyComputed21 = false;
    bool isHomographyComputed23 = false;
    bool isShifted = false;

    clock_t start, end;
    double time_s;
    int frame_count = 0;

    // start clock

    start = clock();
    while(video1.isOpened() && video2.isOpened() && video3.isOpened()){
        bool isFrame1Active = video1.read(Frame1);
        bool isFrame2Active = video2.read(Frame2);
        bool isFrame3Active = video3.read(Frame3);

        cv::imwrite("output_images/input/img1.png", Frame1);
        cv::imwrite("output_images/input/img2.png", Frame2);
        cv::imwrite("output_images/input/img3.png", Frame3);
        // converting frame to grayscale
        // cv::cvtColor(Frame1, Frame1, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(Frame2, Frame2, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(Frame3, Frame3, cv::COLOR_BGR2GRAY);

        // shifing the reference image to the center

        Frame2 = shiftImage(Frame2, 300, 300);
        cv::imwrite("output_images/tranformations/tansformedframe2.png", Frame2);

        // keypoints and descriptors
        std::vector<cv::KeyPoint> kp_vid1 , kp_vid2, kp_vid3;
        cv::Mat des_vid1, des_vid2, des_vid3;

        kp_vid1 = genrateKeyPoints(Frame1, kp_vid1);
        kp_vid2 = genrateKeyPoints(Frame2, kp_vid2);
        kp_vid3 = genrateKeyPoints(Frame3, kp_vid3);

        cv::Mat drawkp_vid1;
        cv::Mat drawkp_vid2;
        cv::Mat drawkp_vid3;

        cv::drawKeypoints(Frame1, kp_vid1, drawkp_vid1);
        cv::drawKeypoints(Frame2, kp_vid2, drawkp_vid2);
        cv::drawKeypoints(Frame3, kp_vid3, drawkp_vid3);

        cv::imwrite("output_images/keypoints/vid1_kp.png", drawkp_vid1);
        cv::imwrite("output_images/keypoints/vid2_kp.png", drawkp_vid2);
        cv::imwrite("output_images/keypoints/vid3_kp.png", drawkp_vid3);

        des_vid1 = genrateDescriptors(Frame1, kp_vid1);
        des_vid2 = genrateDescriptors(Frame2, kp_vid2);
        des_vid3 = genrateDescriptors(Frame3, kp_vid3);

        // converting it to opencv descriptor format sift
        des_vid1.convertTo(des_vid1, CV_32F);
        des_vid2.convertTo(des_vid2, CV_32F);
        des_vid3.convertTo(des_vid3, CV_32F);

        // performing matching
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> rawMatches21;
        std::vector<std::vector<cv::DMatch>> rawMatches23;

        matcher.knnMatch(des_vid2, des_vid1, rawMatches21, 2);
        matcher.knnMatch(des_vid2, des_vid3, rawMatches23, 2);

        cv::Mat rawMatches21_display;
        cv::Mat rawMatches23_display;

        cv::drawMatches(Frame1, kp_vid1, Frame2, kp_vid2, rawMatches21,rawMatches21_display);
        cv::drawMatches(Frame3, kp_vid3, Frame2, kp_vid2, rawMatches23, rawMatches23_display);

        cv::imwrite("output_images/matching/rawMatch21.png", rawMatches21_display);
        cv::imwrite("output_images/matching/rawMatch23.png", rawMatches23_display);


        std::vector<cv::DMatch> goodMatches21, goodMatches23;
        goodMatches21 = LowesRatioClean(rawMatches21);
        goodMatches23 = LowesRatioClean(rawMatches23);

        cv::Mat goodMatches21_display;
        cv::Mat goodMatches23_display;
        cv::drawMatches(Frame2, kp_vid2, Frame1, kp_vid1, goodMatches21, goodMatches21_display);
        cv::drawMatches(Frame2, kp_vid2, Frame3, kp_vid3, goodMatches23, goodMatches23_display);
        cv::imwrite("output_images/matching/goodMatches21.png", goodMatches21_display);
        cv::imwrite("output_images/matching/goodMatches23.png", goodMatches23_display);
        // cv::imshow("matches 23", goodMatches21_display);
        std::vector<cv::Point2f> good_kp12, good_kp21, good_kp32, good_kp23;
        for(auto match23 : goodMatches23){
            good_kp32.push_back(kp_vid2[match23.queryIdx].pt);
            good_kp23.push_back(kp_vid3[match23.trainIdx].pt);
        }

        for(auto match21 : goodMatches21){
            good_kp21.push_back(kp_vid2[match21.queryIdx].pt);
            good_kp12.push_back(kp_vid1[match21.trainIdx].pt);
        }

        if(!isHomographyComputed23){
            cv::Mat Homography23 = cv::findHomography(good_kp23, good_kp32, cv::RANSAC);
            cv::warpPerspective(Frame3, Tranformation23, Homography23, Frame2.size());
            cv::imwrite("output_images/tranformations/Tranfomation23.png", Tranformation23);
            isHomographyComputed23 = true;
        }

        if(!isHomographyComputed21){
            cv::Mat Homography21 = cv::findHomography(good_kp12, good_kp21, cv::RANSAC);
            cv::warpPerspective(Frame1, Tranformation21, Homography21, Frame2.size());
            cv::imwrite("output_images/tranformations/Tranfomation21.png", Tranformation21);
            isHomographyComputed21 = true;
        }
        // loop back video
        // if(!isFrame1Active){
        //     video1.set(cv::CAP_PROP_POS_FRAMES, 0);
        //     continue;
        // }
        // if(!isFrame2Active){
        //     video2.set(cv::CAP_PROP_POS_FRAMES, 0);
        //     continue;
        // }
        // if(!isFrame3Active){
        //     video3.set(cv::CAP_PROP_POS_FRAMES, 0);
        //     continue;
        // }

        // displaying the videos
        // cv::imshow("Video1", Frame1);
        // cv::imshow("Video2", Frame2);
        // cv::imshow("Video3", Frame3);

        // panormoic result
        frame_count++; 
        
        cv::Mat temp_res, result;
        cv::add(Tranformation21, Frame2, temp_res);
        cv::add(temp_res, Tranformation23, result);
        end = clock();
        time_s = ((double) (end - start) / CLOCKS_PER_SEC);
        // cv::putText(res,
        //             "FPS: " + std::to_string(static_cast<int>(
        //                           1.0 / (cv::getTickCount() - prev_frame_time) *
        //                           cv::getTickFrequency())),
        //             cv::Point(7, 70), cv::FONT_HERSHEY_SIMPLEX, 3,
        //             cv::Scalar(100, 255, 0), 3, cv::LINE_AA);
        if(time_s >= 1.0){
            double fps = (double) (frame_count) / time_s;
            std::cout << fps << "\n";
            cv::putText(result,
                        "FPS: " + std::to_string(fps),
                        cv::Point(7, 70), cv::FONT_HERSHEY_SIMPLEX, 3,
                        cv::Scalar(100, 255, 0), 3, cv::LINE_AA);
        }
        cv::imshow("result", result);
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