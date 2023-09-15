import cv2

# Load the image using regular OpenCV imread
img = cv2.imread('img.jpg')
print(cv2.cuda.DeviceInfo(0))
# Create a GPU matrix object and upload the image
cuda_img = cv2.cuda_GpuMat()
cuda_img.upload(img)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey()