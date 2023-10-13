#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
int main(int argc, char const *argv[])
{
    cv::VideoCapture cap1(7);
    cv::VideoCapture cap2(5);

    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::cuda::GpuMat Frame1_gpu;
    cv::cuda::GpuMat Frame2_gpu;
    bool running = true;
    std::cout << "am i here?\n";
    bool homography12 = false;
    cv::Mat TransformationMatrix;
    while (cap1.isOpened() && cap2.isOpened())
    {
        cap1.read(Frame1);
        cap2.read(Frame2);
        cv::cvtColor(Frame1, Frame1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(Frame2, Frame2, cv::COLOR_BGR2GRAY);

        Frame1_gpu.upload(Frame1);
        Frame2_gpu.upload(Frame2);
        // Download the image from GPU to CPU.

        cv::Ptr<cv::cuda::ORB> orb_gpu = cv::cuda::ORB::create(); 
        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::KeyPoint> kp2;
        cv::cuda::GpuMat des1_gpu;
        cv::cuda::GpuMat des2_gpu;

        orb_gpu->detectAndCompute(Frame1_gpu,cv::cuda::GpuMat(),kp1,des1_gpu);
        orb_gpu->detectAndCompute(Frame2_gpu, cv::cuda::GpuMat(), kp2, des2_gpu);

        cv::Mat Frame1_kps;
        cv::Mat Frame2_kps;
        // displaying image of keypoints
        cv::drawKeypoints(Frame1, kp1, Frame1_kps);
        cv::drawKeypoints(Frame2, kp2, Frame2_kps);
        cv::imshow("left keypoints", Frame1_kps);
        cv::imshow("right keypoints", Frame2_kps);


        // matching
        std::vector<cv::DMatch> knn_matches;
        cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);


        matcher->match(des1_gpu, des2_gpu, knn_matches);

        // soting the best points (as lowes ratios test dosent seem to work for this!)
        std::sort(knn_matches.begin(), knn_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b){
                    return a.distance < b.distance;
        });

        int number_of_matches = 50;
        std::vector good_matches(knn_matches.begin(), knn_matches.begin() + number_of_matches);

        // displaying the good keypoints and match
        cv::Mat matchedRes;
        cv::drawMatches(Frame1, kp1, Frame2, kp2, good_matches, matchedRes);
        cv::imwrite("good_matches.png", matchedRes);
        cv::imshow("good matches", matchedRes);
        // storing the keypoints of the matches
        std::vector<cv::Point2f> good_kp1, good_kp2; 
        for(auto match : good_matches){
            good_kp1.push_back(kp1[match.queryIdx].pt);
            good_kp2.push_back(kp2[match.trainIdx].pt);
            std::cout << kp1[match.queryIdx].pt << "\n";
        }

        // genrating homography

        if(!homography12){
            TransformationMatrix = cv::findHomography(good_kp2, good_kp1, cv::RANSAC);
            homography12 = true;
        }

        // performing the tranformaton in GPU
        cv::cuda::GpuMat TransormedFrameFrame1_GPU;
        cv::cuda::warpPerspective(Frame2_gpu, TransormedFrameFrame1_GPU, TransformationMatrix, Frame2.size() + Frame1.size());
        cv::Mat TransormedFrameFrame2;
        // extracting image from GPU to CPU
        TransormedFrameFrame1_GPU.download(TransormedFrameFrame2);
        cv::imwrite("tranformed_image2.png", TransormedFrameFrame2);

        //tranforming train image to a bigger mask of the tranformation matrix
        cv::Mat mask = cv::Mat::zeros(Frame1.size() + Frame2.size(), CV_8U);
        cv::Rect region_of_intrest(0,0, Frame1.cols, Frame1.rows);
        Frame1.copyTo(mask(region_of_intrest));
        cv::Mat TranformedFrameFrame1 = mask;
        cv::imshow("Frame1", mask);

        //perform image addition

        cv::Mat result;
        cv::add(TranformedFrameFrame1, TransormedFrameFrame2, result);
        cv::imshow("result", result);
        cv::imwrite("result.png", result);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' key or Esc key (27) to exit
            break;
        }
    }     
    return 0;
}