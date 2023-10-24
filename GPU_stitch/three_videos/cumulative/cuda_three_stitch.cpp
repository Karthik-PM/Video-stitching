#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
// global declarations
cv::Ptr<cv::cuda::ORB> orb_gpu = cv::cuda::ORB::create();
cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
cv::Mat homography1;
cv::Mat homography2;
// function declations

// keypoint detection
std::vector<cv::KeyPoint> genrateKeyPoints(cv::cuda::GpuMat img1_gpu,
                                           std::vector<cv::KeyPoint> kp);

// show keypoints
cv::Mat displayKeypoints(cv::Mat img, std::vector<cv::KeyPoint> keypoints);

// genrate orb descriptor
cv::cuda::GpuMat genrateDescriptors(cv::cuda::GpuMat img1_gpu,
                                    std::vector<cv::KeyPoint> kp);

// compute good matches
std::vector<cv::DMatch> genrateMatches(cv::cuda::GpuMat des1_gpu,
                                       cv::cuda::GpuMat des2_gpu);

// stitching frame and outputing GPU Mat or CV::Mat respectively
cv::cuda::GpuMat stitchFrame_output_gpu(
    cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
    std::vector<cv::DMatch> matches, std::vector<std::vector<cv::KeyPoint>> kps,
    cv::Size_<int> frame_size, int img1_col, int img2_row, int flag);

cv::Mat stitchFrame_output_cpu(cv::cuda::GpuMat img1_gpu,
                               cv::cuda::GpuMat img2_gpu,
                               std::vector<cv::DMatch> matches,
                               std::vector<std::vector<cv::KeyPoint>> kps,
                               cv::Size_<int> frame_size, int img1_col,
                               int img2_row, bool flag);

int main(int argc, char const *argv[]) {
  cv::VideoCapture cap1("inputVideos/left.mp4");
  cv::VideoCapture cap2("inputVideos/middle.mp4");
  cv::VideoCapture cap3("inputVideos/right.mp4");

  cv::Mat Frame1;
  cv::Mat Frame2;
  cv::Mat Frame3;

  cv::cuda::GpuMat Frame1_gpu;
  cv::cuda::GpuMat Frame2_gpu;
  cv::cuda::GpuMat Frame3_gpu;

  int compute_h12 = 0;
  int compute_h32 = 0;

  bool running = true;

  int prev_frame_time = 0;
  int new_frame_time = 0;
  // float fps_float = 0;
  int fps_int = 0;
  std::string fps_string = "";

  while (running) {

    bool isFrame1Active = cap1.read(Frame1);
    bool isFrame2Active = cap2.read(Frame2);
    bool isFrame3Active = cap3.read(Frame3);

    // loop back video
    if (!isFrame1Active) {
      cap1.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    if (!isFrame2Active) {
      cap2.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    if (!isFrame3Active) {
      cap3.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }

    // converting frame to grayscale
    cv::cvtColor(Frame1, Frame1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Frame2, Frame2, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Frame3, Frame3, cv::COLOR_BGR2GRAY);

    Frame1_gpu.upload(Frame1);
    Frame2_gpu.upload(Frame2);
    Frame3_gpu.upload(Frame3);

    // genrate keypoints
    std::vector<cv::KeyPoint> kp1 = genrateKeyPoints(Frame1_gpu, {});
    std::vector<cv::KeyPoint> kp2 = genrateKeyPoints(Frame2_gpu, {});
    std::vector<cv::KeyPoint> kp3 = genrateKeyPoints(Frame3_gpu, {});

    // genrate descriptors
    cv::cuda::GpuMat des1 = genrateDescriptors(Frame1_gpu, kp1);
    cv::cuda::GpuMat des2 = genrateDescriptors(Frame2_gpu, kp2);
    cv::cuda::GpuMat des3 = genrateDescriptors(Frame3_gpu, kp3);

    // matching phase 1
    std::vector<cv::DMatch> match1 = genrateMatches(des1, des2);

    cv::Mat displayMatches;
    cv::drawMatches(Frame1, kp1, Frame2, kp2, match1, displayMatches);
    // cv::imshow("goodMatches", displayMatches);
    cv::Size_<int> pano1_size = Frame1.size() + Frame2.size();

    // phase 1 stitching
    cv::cuda::GpuMat pano1 = stitchFrame_output_gpu(
        Frame1_gpu, Frame2_gpu, match1, {kp1, kp2}, pano1_size, Frame1.cols,
        Frame1.rows, compute_h12++);

    cv::Mat tranformedFrame;
    pano1.download(tranformedFrame);
    // computing homography once

    // matching phase 2

    std::vector<cv::KeyPoint> pano1_kp = genrateKeyPoints(pano1, {});
    cv::cuda::GpuMat pano_des = genrateDescriptors(pano1, pano1_kp);
    std::vector<cv::DMatch> match2 = genrateMatches(pano_des, des3);

    cv::Mat panocpu;
    pano1.download(panocpu);

    cv::Mat displayMatches1;
    cv::drawMatches(panocpu, pano1_kp, Frame3, kp3, match2, displayMatches1);
    cv::imshow("matches2", displayMatches1);
    cv::imwrite("matches2.png", displayMatches1);

    cv::Mat result;

    cv::Size_<int> result_size;
    result_size.height = pano1_size.height;
    result_size.width = pano1_size.width + Frame3.size().width;
    result = stitchFrame_output_cpu(pano1, Frame3_gpu, match2, {pano1_kp, kp3},
                                    result_size, pano1_size.width,
                                    pano1_size.height, compute_h32++);

    cv::putText(result,
                "FPS: " + std::to_string(static_cast<int>(
                              1.0 / (cv::getTickCount() - prev_frame_time) *
                              cv::getTickFrequency())),
                cv::Point(7, 70), cv::FONT_HERSHEY_SIMPLEX, 3,
                cv::Scalar(100, 255, 0), 3, cv::LINE_AA);
    // display keypoints
    // cv::imshow("kp1", displayKeypoints(Frame1, kp1));
    // cv::imshow("kp2", displayKeypoints(Frame2, kp2));
    // cv::imshow("kp3", displayKeypoints(Frame3, kp3));

    // display input video feed
    // cv::imshow("Frame1", Frame1);
    // cv::imshow("Frame2", Frame2);
    // cv::imshow("Frame3", Frame3);

    cv::imshow("tranformedFrame", tranformedFrame);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) { // 'q' key or Esc key (27) to exit
      running = false;
    }
  }
  return 0;
}

// function definitions

std::vector<cv::KeyPoint> genrateKeyPoints(cv::cuda::GpuMat img1_gpu,
                                           std::vector<cv::KeyPoint> kp) {
  orb_gpu->detect(img1_gpu, kp);
  return kp;
}

cv::cuda::GpuMat genrateDescriptors(cv::cuda::GpuMat img1_gpu,
                                    std::vector<cv::KeyPoint> kp) {
  cv::cuda::GpuMat des;
  orb_gpu->compute(img1_gpu, kp, des);
  return des;
}

cv::Mat displayKeypoints(cv::Mat img, std::vector<cv::KeyPoint> keypoints) {
  cv::Mat keypoint_display;
  cv::drawKeypoints(img, keypoints, keypoint_display);
  return keypoint_display;
}

std::vector<cv::DMatch> genrateMatches(cv::cuda::GpuMat des1_gpu,
                                       cv::cuda::GpuMat des2_gpu) {
  std::vector<cv::DMatch> matches;
  matcher->match(des1_gpu, des2_gpu, matches);
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) {
              return a.distance < b.distance;
            });

  int number_of_matches = 50;
  std::vector good_matches(matches.begin(),
                           matches.begin() + number_of_matches);
  return good_matches;
}

cv::cuda::GpuMat stitchFrame_output_gpu(
    cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
    std::vector<cv::DMatch> matches, std::vector<std::vector<cv::KeyPoint>> kps,
    cv::Size_<int> frame_size, int img1_col, int img1_row, int flag) {

  // storing good keypoints
  cv::cuda::GpuMat transformedFrameRight;
  cv::cuda::GpuMat result;
  std::vector<cv::Point2f> good_kp1, good_kp2;
  for (auto match : matches) {
    good_kp1.push_back(kps[0][match.queryIdx].pt);
    good_kp2.push_back(kps[1][match.trainIdx].pt);
  }

  if (flag < 10) {
    homography1 = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);
  }
  // tranforming train image to a bigger mask of the tranformation matrix

  cv::cuda::warpPerspective(img2_gpu, transformedFrameRight, homography1,
                            frame_size);
  cv::Mat transformedFrameRight_cpu;

  // extracting image from GPU to CPU
  transformedFrameRight.download(transformedFrameRight_cpu);
  cv::imwrite("tranformed_image2.png", transformedFrameRight_cpu);

  cv::Mat mask = cv::Mat::zeros(frame_size, CV_8U);
  cv::Rect region_of_intrest(0, 0, img1_col, img1_row);
  cv::Mat img1;
  img1_gpu.download(img1);
  img1.copyTo(mask(region_of_intrest));
  cv::Mat TranformedFrameFrame1 = mask;
  cv::imshow("Frame1", mask);
  // perform image addition
  cv::imwrite("TranformedFrameRight.png", TranformedFrameFrame1);
  cv::Mat result_cpu;
  cv::add(TranformedFrameFrame1, transformedFrameRight_cpu, result_cpu);
  result.upload(result_cpu);
  return result;
}

cv::Mat stitchFrame_output_cpu(cv::cuda::GpuMat img1_gpu,
                               cv::cuda::GpuMat img2_gpu,
                               std::vector<cv::DMatch> matches,
                               std::vector<std::vector<cv::KeyPoint>> kps,
                               cv::Size_<int> frame_size, int img1_col,
                               int img1_row, bool flag) {
  // storing good keypoints
  cv::cuda::GpuMat transformedFrameRight;
  cv::cuda::GpuMat result;
  std::vector<cv::Point2f> good_kp1, good_kp2;
  for (auto match : matches) {
    good_kp1.push_back(kps[0][match.queryIdx].pt);
    good_kp2.push_back(kps[1][match.trainIdx].pt);
  }

  if (flag < 1) {
    homography2 = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);
  }
  // tranforming train image to a bigger mask of the tranformation matrix

  cv::cuda::warpPerspective(img2_gpu, transformedFrameRight, homography2,
                            frame_size);
  cv::Mat transformedFrameRight_cpu;

  // extracting image from GPU to CPU
  transformedFrameRight.download(transformedFrameRight_cpu);
  cv::imwrite("tranformed_image2.png", transformedFrameRight_cpu);

  cv::Mat mask = cv::Mat::zeros(frame_size, CV_8U);
  cv::Rect region_of_intrest(0, 0, img1_col, img1_row);
  cv::Mat img1;
  img1_gpu.download(img1);
  img1.copyTo(mask(region_of_intrest));
  cv::Mat TranformedFrameFrame1 = mask;
  // cv::imshow("Frame1", mask);
  // perform image addition
  cv::Mat result_cpu;
  cv::add(TranformedFrameFrame1, transformedFrameRight_cpu, result_cpu);
  return result_cpu;
}
