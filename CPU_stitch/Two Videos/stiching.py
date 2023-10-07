import cv2
import numpy as np
def main():
    left = cv2.VideoCapture(3)
    middle = cv2.VideoCapture(5)
    # left = cv2.VideoCapture(2);
    # middle = cv2.VideoCapture(4);
    # print(middle.isOpened())
    # print(left)
    print(left.isOpened() , middle.isOpened())
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    HomoGraphyFlag = True
    ExceptionFlag = True
    while(left.isOpened() and middle.isOpened()):
        # print("hello")
        ret, framemiddle = middle.read()
        ret, frameLeft = left.read()
        frameLeftGray = cv2.cvtColor(frameLeft, cv2.COLOR_RGB2GRAY)
        framemiddleGray = cv2.cvtColor(framemiddle, cv2.COLOR_RGB2GRAY)
        kp1, des1 = sift.detectAndCompute(framemiddle, None)
        kp2, des2 = sift.detectAndCompute(frameLeft, None)
        featuresf1 = cv2.drawKeypoints(
            framemiddle, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        featuresf2 = cv2.drawKeypoints(
            frameLeft, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("middle", featuresf1)
        cv2.imshow("left", featuresf2)
        rawMatches = bf.knnMatch(des1, des2, k=2)
        # drawMatches = cv2.drawMatchesKnn(frameLeft , kp1 , framemiddle , kp2 , rawMatches , None , flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        good_points = []
        good_matches = []
        for m1, m2 in rawMatches:
            # performing the Lowe's ratios test for getting the nearest points ie the good points
            if m1.distance < 0.85 * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        # draws the matching keypoints btw the two images
        drawMatches = cv2.drawMatchesKnn(
            framemiddle, kp1, frameLeft, kp2, good_matches, None, flags=2)
        if len(good_points) > 10:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            if(HomoGraphyFlag):
                H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
                print(H)
                HomoGraphyFlag = False

        TransformedFrameLeft = cv2.warpPerspective(frameLeft, H, (frameLeft.shape[1] + framemiddle.shape[1], framemiddle.shape[0]))
        cv2.imshow("good Matches", drawMatches)
        # cv2.cvtColor(cv2.cvtColor(framemiddle , cv2.COLOR_GRAY2RGB) , cv2.COLOR_RGB2GRAY)
        TranformedFramemiddle = np.zeros([TransformedFrameLeft.shape[0], TransformedFrameLeft.shape[1] , 3], dtype = np.uint8)
        TranformedFramemiddle[:framemiddle.shape[0],:framemiddle.shape[1]] = framemiddle
        TransformedFrameLeft[:framemiddle.shape[0],:framemiddle.shape[1]] = framemiddle
        # print(TranformedFramemiddle.shape ,":", TransformedFrameLeft.shape)
        # cv2.imshow("Transformedmiddle", TranformedFramemiddle)
        # cv2.imshow("Framemiddle" , framemiddle)
        
        try:
            # sitchedImg = cv2.add(TransformedFrameLeft , TranformedFramemiddle)
            cv2.imshow("Left" , TranformedFramemiddle)
            # cv2.imshow("right" , TransformedFrameLeft)
            # cv2.imshow("panoramic image " , sitchedImg)
        except Exception as e:
            if(ExceptionFlag):
                print(e)
                ExceptionFlag = False
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
if __name__ == "__main__":
    main()
