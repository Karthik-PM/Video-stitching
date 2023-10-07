#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

int main() {
    cv::Mat img1 = cv::imread("images/left.png");
    cv::Mat img2 = cv::imread("images/right.png");
    cv::Mat img1_gray;
    cv::Mat img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat img1_gpu;
    cv::cuda::GpuMat img2_gpu;
    img1_gpu.upload(img1_gray);
    img2_gpu.upload(img2_gray);

    // Download the image from GPU to CPU.
    cv::Ptr<cv::cuda::ORB> orb_gpu = cv::cuda::ORB::create(); 
    // cv::Ptr<cv::cuda::SURF_CUDA> surf_cuda = cv::cuda::SURF_CUDA()::create();
    std::vector<cv::KeyPoint> kp1;
    std::vector<cv::KeyPoint> kp2;
    cv::cuda::GpuMat des1_gpu;
    cv::cuda::GpuMat des2_gpu;

    orb_gpu->detectAndCompute(img1_gpu,cv::cuda::GpuMat(),kp1,des1_gpu);
    orb_gpu->detectAndCompute(img2_gpu, cv::cuda::GpuMat(), kp2, des2_gpu);
    cv::Mat img1_kps;
    cv::Mat img2_kps;

    // displaying image of keypoints
    cv::drawKeypoints(img1, kp1, img1_kps);
    cv::drawKeypoints(img2, kp2, img2_kps);
    cv::imshow("left keypoints", img1_kps);
    cv::imshow("right keypoints",img2_kps);

    // matching
    std::vector<cv::DMatch> knn_matches;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    matcher->match(des1_gpu, des2_gpu, knn_matches);

    // soting the best points (as lowes ratios test dosent seem to work for this!)
    std::sort(knn_matches.begin(), knn_matches.end());
    int number_of_matches = 50;
    std::vector good_matches(knn_matches.begin(), knn_matches.begin() + number_of_matches);

    // displaying the good keypoints and match
    cv::Mat matchedRes;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, matchedRes);
    cv::imwrite("good_matches.png", matchedRes);

    // storing the keypoints of the matches
    std::vector<cv::Point2f> good_kp1, good_kp2; 
    for(auto match : good_matches){
        good_kp1.push_back(kp1[match.queryIdx].pt);
        good_kp2.push_back(kp2[match.trainIdx].pt);
        std::cout << kp1[match.queryIdx].pt << "\n";
    }

    // genrating homography
    cv::Mat TransformationMatrix = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);

    // performing the tranformaton in GPU
    cv::cuda::GpuMat TransormedFrameimg1_GPU;
    cv::cuda::warpPerspective(img2_gpu, TransormedFrameimg1_GPU, TransformationMatrix, img2.size() + img1.size());
    cv::Mat TransormedFrameimg2;
    // extracting image from GPU to CPU
    TransormedFrameimg1_GPU.download(TransormedFrameimg2);
    cv::imwrite("tranformed_image2.png", TransormedFrameimg2);

    //tranforming train image to a bigger mask of the tranformation matrix
    cv::Mat mask = cv::Mat::zeros(img1.size() + img2.size(), CV_8U);
    cv::Rect region_of_intrest(0,0, img1.cols, img1.rows);
    img1_gray.copyTo(mask(region_of_intrest));
    cv::Mat TranformedFrameimg1 = mask;
    // perform image addition
    cv::Mat result;
    cv::add(TranformedFrameimg1, TransormedFrameimg2, result);
    cv::imshow("result", result);
    cv::imwrite("result.png", result);
    cv::waitKey(1000);

    return 0;
}
