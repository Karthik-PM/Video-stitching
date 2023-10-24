#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <time.h>

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
cv::cuda::GpuMat
stitchFrame_output_gpu1(cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
                        std::vector<cv::DMatch> matches,
                        std::vector<std::vector<cv::KeyPoint>> kps,
                        cv::Size_<int> frame_size, int flag);

cv::cuda::GpuMat
stitchFrame_output_gpu2(cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
                        std::vector<cv::DMatch> matches,
                        std::vector<std::vector<cv::KeyPoint>> kps,
                        cv::Size_<int> frame_size, int flag);

cv::Mat stitchFrame_output_cpu(cv::cuda::GpuMat img1_gpu,
                               cv::cuda::GpuMat img2_gpu,
                               std::vector<cv::DMatch> matches,
                               std::vector<std::vector<cv::KeyPoint>> kps,
                               cv::Size_<int> frame_size, int img1_col,
                               int img2_row, bool flag);

// shifting image to center
cv::cuda::GpuMat shiftFrame(cv::cuda::GpuMat img1_gpu, int x, int y);

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
  double prev_frame_time = 0;

  // recording output
  cv::VideoWriter writer;
  int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
  std::string fileName = "three_stitch.mp4";
  cv::Size frameOutput(1080, 500);
  double fps = 55.0;
  writer.open(fileName, codec, fps, frameOutput, false);

  clock_t start, end;
  double time_s;
  int frame_count = 0;

  // start clock

  start = clock();
  while (running) {

    bool isFrame1Active = cap1.read(Frame1);
    bool isFrame2Active = cap2.read(Frame2);
    bool isFrame3Active = cap3.read(Frame3);

    // loop back video
    // if (!isFrame1Active) {
    //   cap1.set(cv::CAP_PROP_POS_FRAMES, 0);
    //   continue;
    // }
    // if (!isFrame2Active) {
    //   cap2.set(cv::CAP_PROP_POS_FRAMES, 0);
    //   continue;
    // }
    // if (!isFrame3Active) {
    //   cap3.set(cv::CAP_PROP_POS_FRAMES, 0);
    //   continue;
    // }

    // converting frame to grayscale
    cv::cvtColor(Frame1, Frame1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Frame2, Frame2, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Frame3, Frame3, cv::COLOR_BGR2GRAY);

    // upload frames to GPU
    Frame1_gpu.upload(Frame1);
    Frame2_gpu.upload(Frame2);
    Frame3_gpu.upload(Frame3);

    // shift middle frame by 300 units x and y axis
    Frame2_gpu = shiftFrame(Frame2_gpu, 300, 300);

    // genrate keypoints
    std::vector<cv::KeyPoint> kp1 = genrateKeyPoints(Frame1_gpu, {});
    std::vector<cv::KeyPoint> kp2 = genrateKeyPoints(Frame2_gpu, {});
    std::vector<cv::KeyPoint> kp3 = genrateKeyPoints(Frame3_gpu, {});

    // genrate descriptors
    cv::cuda::GpuMat des1 = genrateDescriptors(Frame1_gpu, kp1);
    cv::cuda::GpuMat des2 = genrateDescriptors(Frame2_gpu, kp2);
    cv::cuda::GpuMat des3 = genrateDescriptors(Frame3_gpu, kp3);

    std::vector<cv::DMatch> match1 = genrateMatches(des2, des1);
    std::vector<cv::DMatch> match2 = genrateMatches(des2, des3);

    cv::Size_<int> mask_size = Frame2.size() * 2;

    // stitch Frame 1 and Frame 2
    cv::cuda::GpuMat leftFrame = stitchFrame_output_gpu1(
        Frame2_gpu, Frame1_gpu, match1, {kp2, kp1}, mask_size, compute_h12++);

    // stitch Frame 2 and Frame 2
    cv::cuda::GpuMat rightFrame = stitchFrame_output_gpu2(
        Frame2_gpu, Frame3_gpu, match2, {kp2, kp3}, mask_size, compute_h32++);

    cv::Mat leftFramecpu;
    cv::Mat rightFramecpu;

    // dowloading warpped results to CPU
    leftFrame.download(leftFramecpu);
    rightFrame.download(rightFramecpu);

    // cv::imshow("leftFrame", leftFramecpu);
    // cv::imshow("rightFrame", rightFramecpu);
    Frame2_gpu.download(Frame2);

    cv::Mat res;
    cv::add(leftFramecpu, rightFramecpu, res);
    cv::imwrite("tranfromations.png", res);

    // adding the tranformed images and genrating results
    cv::add(res, Frame2, res);

    // frame count 
    frame_count ++;

    // cropping the result from the big mask
    cv::Rect ROI(0, 300, mask_size.width - 200, 500);
    res = res(ROI);

    // end time
    end = clock();
    time_s = ((double) (end - start) / CLOCKS_PER_SEC);

    // frame rate estimation
    if(time_s >= 1.0){
        double fps = static_cast<double> (frame_count) / time_s;
        cv::putText(res,
                    "FPS: " + std::to_string(static_cast<int> (fps)),
                    cv::Point(7, 70), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(100, 255, 0), 3, cv::LINE_AA);
    }

    // display result
    cv::imshow("res", res);

    if(compute_h12 >= 12){
      writer.write(res);
    }
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) { // 'q' key or Esc key (27) to exit
      running = false;
    }
  }
  cap1.release();
  cap2.release();
  cap3.release();

  writer.release();
  return 0;
}

// function definitions

cv::cuda::GpuMat shiftFrame(cv::cuda::GpuMat image, int x, int y) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
  cv::cuda::warpPerspective(image, image, shift, image.size() * 2);
  return image;
}

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

cv::cuda::GpuMat
stitchFrame_output_gpu2(cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
                        std::vector<cv::DMatch> matches,
                        std::vector<std::vector<cv::KeyPoint>> kps,
                        cv::Size_<int> frame_size, int flag) {

  // storing good keypoints
  cv::cuda::GpuMat transformedFrameRight;
  cv::cuda::GpuMat result;

  std::vector<cv::Point2f> good_kp1, good_kp2;
  for (auto match : matches) {
    good_kp1.push_back(kps[0][match.queryIdx].pt);
    good_kp2.push_back(kps[1][match.trainIdx].pt);
  }

  if (flag < 1) {
    homography1 = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);
  }
  std::cout << flag << "\n";
  // tranforming train image to a bigger mask of the tranformation matrix

  cv::cuda::warpPerspective(img2_gpu, transformedFrameRight, homography1,
                            frame_size);
  cv::Mat transformedFrameRight_cpu;

  // extracting image from GPU to CPU
  transformedFrameRight.download(transformedFrameRight_cpu);
  cv::imwrite("tranformed_image2.png", transformedFrameRight_cpu);
  // perform image addition
  return transformedFrameRight;
}

cv::cuda::GpuMat
stitchFrame_output_gpu1(cv::cuda::GpuMat img1_gpu, cv::cuda::GpuMat img2_gpu,
                        std::vector<cv::DMatch> matches,
                        std::vector<std::vector<cv::KeyPoint>> kps,
                        cv::Size_<int> frame_size, int flag) {

  // storing good keypoints
  cv::cuda::GpuMat transformedFrameRight;
  cv::cuda::GpuMat result;
  std::vector<cv::Point2f> good_kp1, good_kp2;
  for (auto match : matches) {
    good_kp1.push_back(kps[0][match.queryIdx].pt);
    good_kp2.push_back(kps[1][match.trainIdx].pt);
  }

  if (flag <= 12) {
    homography2 = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);
  }
  // std::cout << flag << "\n";
  // tranforming train image to a bigger mask of the tranformation matrix

  cv::cuda::warpPerspective(img2_gpu, transformedFrameRight, homography2,
                            frame_size);

  cv::Mat transformedFrameRight_cpu;

  // extracting image from GPU to CPU
  transformedFrameRight.download(transformedFrameRight_cpu);
  cv::imwrite("tranformed_image2.png", transformedFrameRight_cpu);
  // perform image addition
  return transformedFrameRight;
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
  cv::imshow("Frame1", mask);

  // perform image addition
  cv::Mat result_cpu;
  cv::add(TranformedFrameFrame1, transformedFrameRight_cpu, result_cpu);
  return result_cpu;

}