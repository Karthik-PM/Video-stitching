#include <bits/stdc++.h>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>
// globlal declaration
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

std::vector<cv::KeyPoint>
genrateKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint> keypoints) {
  sift->detect(frame, keypoints);
  return keypoints;
}

cv::Mat genrateDescriptors(cv::Mat frame, std::vector<cv::KeyPoint> keypoints) {
  cv::Mat descriptor;
  sift->compute(frame, keypoints, descriptor);
  return descriptor;
}

cv::Mat shiftImage(cv::Mat image, int x, int y) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
  cv::warpPerspective(image, image, shift, image.size() * 2);
  return image;
}

std::vector<cv::DMatch>
LowesRatioClean(std::vector<std::vector<cv::DMatch>> rawmatches) {
  std::vector<cv::DMatch> goodMatches;
  double ratio = 0.8;
  for (auto match : rawmatches) {
    // std::cout << match[0].distance << " " << match[1].distance << "\n";
    if (match[0].distance < ratio * match[1].distance) {
      goodMatches.push_back(match[0]);
    }
  }
  return goodMatches;
}
float calculateFrameRate(cv::VideoCapture &video) {
  // Get the start time
  double start_time = cv::getTickCount();

  // Read frames until the end of the video or until the user presses a key
  int num_frames = 0;
  cv::Mat frame;
  while (video.read(frame)) {
    num_frames++;
  }

  // Get the end time
  double end_time = cv::getTickCount();

  // Calculate the elapsed time
  double elapsed_time = (end_time - start_time) / cv::getTickFrequency();

  // Calculate the frame rate
  float frame_rate = num_frames / elapsed_time;

  // Return the frame rate
  return frame_rate;
}

int main(int argc, char const *argv[]) {
  cv::VideoCapture video1(3);
  cv::VideoCapture video2(5);
  // cv::VideoCapture video2("vid2.mp4");
  // cv::VideoCapture video2("vid2.mp4");

  cv::Mat Frame1;
  cv::Mat Frame2;
  cv::Mat result;
  bool isHomographyComputed21 = false;
  // std::cout << "frame rate vid1: " << calculateFrameRate(video1) << "\n";
  // std::cout << "frame rate vid1: " << calculateFrameRate(video2) << "\n";
  while (video1.isOpened() && video2.isOpened()) {

    bool isFrame1Active = video1.read(Frame1);
    bool isFrame2Active = video2.read(Frame2);
    // shifing the reference image to the center

    Frame2 = shiftImage(Frame2, 300, 300);
    cv::imwrite("output_images/tranformations/tansformedframe2.png", Frame2);

    // keypoints and descriptors
    std::vector<cv::KeyPoint> kp_vid1, kp_vid2;
    cv::Mat des_vid1, des_vid2;

    kp_vid1 = genrateKeyPoints(Frame1, kp_vid1);
    kp_vid2 = genrateKeyPoints(Frame2, kp_vid2);

    cv::Mat drawkp_vid1;
    cv::Mat drawkp_vid2;

    cv::drawKeypoints(Frame1, kp_vid1, drawkp_vid1);
    cv::drawKeypoints(Frame2, kp_vid2, drawkp_vid2);

    cv::imwrite("output_images/keypoints/vid1_kp.png", drawkp_vid1);
    cv::imwrite("output_images/keypoints/vid2_kp.png", drawkp_vid2);

    des_vid1 = genrateDescriptors(Frame1, kp_vid1);
    des_vid2 = genrateDescriptors(Frame2, kp_vid2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> rawMatches21;

    des_vid1.convertTo(des_vid1, CV_32F);
    des_vid2.convertTo(des_vid2, CV_32F);
    matcher.knnMatch(des_vid1, des_vid2, rawMatches21, 2);
    cv::Mat rawMatches21_display;

    cv::drawMatches(Frame1, kp_vid1, Frame2, kp_vid2, rawMatches21,
                    rawMatches21_display);

    cv::imwrite("output_images/matching/rawMatch21.png", rawMatches21_display);

    // std::sort(rawMatches21.begin(), rawMatches21.end());
    std::vector<cv::DMatch> goodMatches21 = LowesRatioClean(rawMatches21);

    cv::Mat goodMatches21_display;
    cv::Mat goodMatches23_display;
    cv::drawMatches(Frame1, kp_vid1, Frame2, kp_vid2, goodMatches21,
                    goodMatches21_display);
    cv::imwrite("output_images/matching/goodMatches21.png",
                goodMatches21_display);
    std::vector<cv::Point2f> good_kp12, good_kp21;

    // cv::imshow("matches",goodMatches21_display);
    for (auto match21 : goodMatches21) {
      good_kp21.push_back(kp_vid1[match21.queryIdx].pt);
      good_kp12.push_back(kp_vid2[match21.trainIdx].pt);
    }
    cv::Mat Homography21 = cv::findHomography(good_kp21, good_kp12, cv::RANSAC);

    cv::Mat Tranformation21;
    if (!isHomographyComputed21) {
      cv::warpPerspective(Frame1, Tranformation21, Homography21, Frame2.size());
      cv::imwrite("output_images/tranformations/Tranfomation21.png",
                  Tranformation21);
      isHomographyComputed21 = true;
    }
    // loop back video
    if (!isFrame1Active) {
      video1.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    if (!isFrame2Active) {
      video2.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    // displaying the videos
    cv::imshow("Video1", Frame1);
    cv::imshow("Video2", Frame2);
    // cv::resize(Tranformation21, Tranformation21, Frame2.size());
    // cv::add(Frame2, Tranformation21, result);
    // cv::imshow("result", result);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) { // 'q' key or Esc key (27) to exit
      break;
    }
  }
  video1.release();
  video1.release();
  cv::destroyAllWindows();
  return 0;
}
