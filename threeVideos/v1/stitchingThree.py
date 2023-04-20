import cv2
import numpy as np 
import subprocess
class VideoStiching:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1 , crossCheck = False)
    def KeypointsAndDescriptors(self , img):
        return self.sift.detectAndCompute(img , None)

    def PrintKeypoints(self, img , keypoints):
        keypoints = cv2.drawKeypoints(img , keypoints , None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints" , keypoints)

    def matchDescriptors(self,srcDis , desDis):
        rawMatches = self.bf.knnMatch(srcDis , desDis , k = 2)
        goodKeypoints = []
        goodMatches = []
        for m1 , m2 in rawMatches:
            if m1.distance < 0.8 * m2.distance:
                goodMatches.append([m1])
                goodKeypoints.append((m1.trainIdx , m1.queryIdx))
        return goodMatches , goodKeypoints

    def drawMatches(self,img1 , kp1 , img2 , kp2 , matches):
        img = cv2.drawMatchesKnn(img1 , kp1 , img2 , kp2 , matches , None)
        cv2.imshow("matches" , img)
    
    def DrawPanorama(self , vid1 , vid2 , maskSize):
        computeHomographyOnce = True
        while(vid1.isOpened() and vid2.isOpened()):
            _ , vid1Frame = vid1.read()
            _ , vid2Frame = vid2.read()
            leftKey , leftDes = self.KeypointsAndDescriptors(vid1Frame)
            rightKey , rightDes = self.KeypointsAndDescriptors(vid2Frame)
            m , kp = self.matchDescriptors(rightDes , leftDes)
            # print(kp)
            srcKeypoints = np.array([rightKey[i].pt for (_ , i) in kp])
            desKeypoints = np.array([leftKey[i].pt for (i , _) in kp])
            # self.drawMatches(vid1Frame , srcKeypoints , vid2Frame , desKeypoints ,m)
            # print(srcKeypoints)
            if(computeHomographyOnce):
                if(maskSize == None):
                    maskSize = (vid1Frame.shape[0] + vid2Frame.shape[1] , vid2Frame.shape[1] )
                H , status = cv2.findHomography(np.float32(srcKeypoints) , np.float32(desKeypoints) , cv2.RANSAC , 5.0)
                computeHomographyOnce = False
            # print(maskSize)
            TranformedVid2 = cv2.warpPerspective(vid2Frame , H , maskSize)
            TranformedVid1 = np.zeros([TranformedVid2.shape[0] , TranformedVid2.shape[1] , 3] , np.uint8)
            TranformedVid1[:vid1Frame.shape[0] , :vid1Frame.shape[1]] = vid1Frame
            cv2.imshow("output" , cv2.add(TranformedVid1 , TranformedVid2))

            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

    def DrawPanorama(self , vid1 , vid2 , vid3 , maskSize):
        computeHomographyOnce = True
        while(vid1.isOpened() and vid2.isOpened()):
            _ , vid1Frame = vid1.read()
            _ , vid2Frame = vid2.read()
            _ , vid3Frame = vid3.read()
            leftKey , leftDes = self.KeypointsAndDescriptors(vid1Frame)
            middleKey , middleDes = self.KeypointsAndDescriptors(vid2Frame)
            rightKey , rightDes = self.KeypointsAndDescriptors(vid3Frame)

            m , goodMatchesLeftMid = self.matchDescriptors(middleDes , leftDes)
            m , goodMatchesRightMid = self.matchDescriptors(rightDes , middleDes)
            srcKeypointsLeftMid = np.array([middleKey[i].pt for (_ , i) in goodMatchesLeftMid])
            desKeypointsLeftMid = np.array([leftKey[i].pt for (i , _) in goodMatchesLeftMid])
            srcKeypointsRightMid = np.array([rightKey[i].pt for (_ , i) in goodMatchesRightMid])
            desKeypointsRightMid = np.array([middleKey[i].pt for (i , _ ) in goodMatchesRightMid])
            if(computeHomographyOnce):
                if(maskSize == None):
                    maskSize = (vid1Frame.shape[0] + vid2Frame.shape[0] + vid3Frame.shape[1] , vid2Frame.shape[1])
                leftMiddleTansformation , status = cv2.findHomography(np.float32(srcKeypointsLeftMid) , np.float32(desKeypointsLeftMid) , cv2.RANSAC , 5.0)
                MidRightHomographyTranformation , status = cv2.findHomography(np.float32(srcKeypointsRightMid) , np.float32(desKeypointsRightMid) , cv2.RANSAC , 5.0)
                computeHomographyOnce = False
            # print(maskSize)
            TranformedVid2 = cv2.warpPerspective(vid2Frame , leftMiddleTansformation , maskSize)
            TranformedVid1 = np.zeros([TranformedVid2.shape[0] , TranformedVid2.shape[1] , 3] , np.uint8)
            TranformedVid1[:vid1Frame.shape[0] , :vid1Frame.shape[1]] = vid1Frame
            TranformedVid1Vid2 = cv2.add(TranformedVid1 , TranformedVid2)
            TranformedVid1Vid2 = cv2.warpPerspective(TranformedVid1Vid2 , leftMiddleTansformation , (TranformedVid1Vid2.shape[0] * 5 , vid3Frame.shape[1] * 5))
            TranformedVid3 = np.zeros([TranformedVid1Vid2.shape[0] , TranformedVid1Vid2.shape[1] , 3], np.uint8)
            TranformedVid3[:vid3Frame.shape[0] , :vid3Frame.shape[1]] = vid3Frame
            # cv2.imshow("vid3" , TranformedVid3)
            # TranformedVid1Vid2Vid3 = cv2.add(TranformedVid1Vid2 , TranformedVid3)
            # cv2.imshow("vid1vid2" , TranformedVid1Vid2)
            TranformedVid1Vid2Vid3 = cv2.add(TranformedVid3 , TranformedVid1Vid2)
            cv2.imshow("panorama",TranformedVid1Vid2Vid3)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

def main():
    vid1 = cv2.VideoCapture("middle.mp4")
    vid2 = cv2.VideoCapture("left.mp4")
    vid3 = cv2.VideoCapture("right.mp4")

    stitcher = VideoStiching()
    stitcher.DrawPanorama(vid1 , vid2 , vid3 , None)

if __name__ == "__main__":
    main()