import cv2

# Initialize a list to store camera names
camera_names = []

# Try to open each camera and get its name
for camera_index in range(1, 10):  # Try up to 10 cameras (you can adjust the range)
    camera = cv2.VideoCapture(camera_index)
    if camera.isOpened():
        camera_name = f"Camera {camera_index}"
        camera_names.append((camera_name, camera))
    else:
        camera.release()

# Capture and save a frame from each detected camera
for name, camera in camera_names:
    ret, frame = camera.read()
    if ret:
        filename = f"{name}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved frame from {name} as {filename}")
    camera.release()

# Release the cameras
for _, camera in camera_names:
    camera.release()

