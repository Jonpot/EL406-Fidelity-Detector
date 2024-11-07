import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector as apriltag

def threshold_video_movement(video_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect significant changes in a video using background subtraction and thresholding.

    Args:
        video_path: Path to the video file.

    Returns:
        A thresholded image highlighting significant changes in the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Create a background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=40, detectShadows=False)
    
    # Initialize variables
    accumulated_mask_front = None
    accumulated_mask_back = None
    frame_count = 0
    current_row = 0
    between_nozzles = False
    
    fiducial_coordinate = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if fiducial_coordinate is None:
            fiducial_coordinate = detect_fiducial(frame)

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        # if more than 20% of the frame is moving, this is a significant change, and we break and don't accumulate
        #print(np.count_nonzero(fg_mask))
        #print(np.count_nonzero(fg_mask), 0.1 * fg_mask.size)
        if not between_nozzles and frame_count != 0 and np.count_nonzero(fg_mask) > 0.025 * fg_mask.size:
            if current_row < 2:
                print("BETWEEN NOZZLES")
                between_nozzles = True
            else:
                break

        if between_nozzles and np.count_nonzero(fg_mask) < 0.025 * fg_mask.size:
            print("SWITCHING ROWS")
            between_nozzles = False
            current_row += 1
        
        # Initialize the accumulated mask with the same size as fg_mask
        # Accumulate the foreground masks
        if current_row == 0:
            if accumulated_mask_front is None:
                accumulated_mask_front = np.zeros_like(fg_mask, dtype=np.float32)
            accumulated_mask_front += fg_mask.astype(np.float32)
        elif current_row == 2:
            if accumulated_mask_back is None:
                accumulated_mask_back = np.zeros_like(fg_mask, dtype=np.float32)
            accumulated_mask_back += fg_mask.astype(np.float32)
        
        frame_count += 1

    cap.release()

    # Normalize the accumulated mask
    accumulated_mask_front /= frame_count
    accumulated_mask_back /= frame_count

    # Convert accumulated mask to 8-bit image
    accum_mask_uint8_front = cv2.normalize(accumulated_mask_front, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    accum_mask_uint8_back = cv2.normalize(accumulated_mask_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply adaptive thresholding (Binary Threshold method) to find significant changes
    _, thresholded_image_front_nozzles = cv2.threshold(accum_mask_uint8_front, 0, 255, cv2.THRESH_BINARY)
    _, thresholded_image_back_nozzles = cv2.threshold(accum_mask_uint8_back, 0, 255, cv2.THRESH_BINARY)

    # Return the thresholded image highlighting significant changes
    return thresholded_image_front_nozzles, thresholded_image_back_nozzles, fiducial_coordinate

#output_image = threshold_video_movement('test-clogged.mjpeg')
#output_image = detect_clogged_nozzles('test2_0920.mjpeg')
front_nozzles, back_nozzles = threshold_video_movement('videos/Lighting2-1.mp4')
cv2.imwrite('clogged_nozzles_front.png', front_nozzles)
cv2.imwrite('clogged_nozzles_back.png', back_nozzles)


def detect_fiducial(frame: np.ndarray) -> np.ndarray:
    """
    Detect fiducial in the frame using AprilTag.

    Args:
        frame: Image frame.

    Returns:
        The coordinate of the fiducial.
    """

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = apriltag()
    detector.addFamily("tagStandard41h12")

    detections = detector.detect(img)

    center = None
    for detection in detections:
        if detection.getId() == 2:
            center = detection.getCenter()

    return center