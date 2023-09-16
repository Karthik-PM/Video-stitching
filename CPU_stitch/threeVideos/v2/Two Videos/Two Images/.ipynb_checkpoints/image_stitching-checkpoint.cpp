#include<bits/stdc++.h>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/features2d.hpp>

int main(int argc, char const *argv[])
{
    cv::Mat img1 = cv::imread("left.png");
    cv::Mat img2 = cv::imread("right.png");
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat des1, des2;
    std::vector<cv::KeyPoint> kp1, kp2;
    sift->detect(img1, kp1);
    sift->compute(img1,kp1, des1);
    sift->detect(img2, kp2);
    sift->compute(img2,kp2, des2);
    cv::Mat keypoints1;
    cv::Mat keypoints2;
    cv::drawKeypoints(img1, kp1, keypoints1);
    cv::drawKeypoints(img2, kp2, keypoints2);
    // cv::imshow("keypoints in img1", keypoints1);
    // cv::imshow("keypoints in img2", keypoints2);
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    int k = 2;
    matcher.knnMatch(des1, des2, matches, k);
    double ratio = 0.7;
    std::vector<cv::DMatch> goodmatches;
    std::vector<cv::Point2f> goodkp1, goodkp2;
    for(int i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < matches[i][1].distance * ratio) {
            goodmatches.push_back(matches[i][0]);
            goodkp1.push_back(kp1[matches[i][0].queryIdx].pt);
            goodkp2.push_back(kp1[matches[i][0].trainIdx].pt);
        }
    }
    cv::Mat Homography = cv::findHomography(goodkp1, goodkp2, cv::RANSAC);
    cv::Mat wrapImage;
    cv::warpPerspective(img2, wrapImage, Homography, img1.size());
    cv::Mat mask = cv::Mat::zeros(img1.size(), CV_8U);
    cv::rectangle(mask, cv::Rect(0, 0, img1.cols, img1.rows), cv::Scalar(255), cv::FILLED);

    // cv::Mat stichedImage;
    // img1.copyTo(stichedImage, mask);
    // wrapImage.copyTo(stichedImage, ~mask);
    cv::imshow("Stitched image", wrapImage);
    cv::waitKey(0);
    return 0;
}
