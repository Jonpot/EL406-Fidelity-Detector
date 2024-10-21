import cv2
import numpy as np

def threshold_video_movement(video_path: str) -> np.ndarray:
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
    accumulated_mask = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        # if more than 20% of the frame is moving, this is a significant change, and we break and don't accumulate
        #print(np.count_nonzero(fg_mask))
        if frame_count != 0 and np.count_nonzero(fg_mask) > 0.1 * fg_mask.size:
            break
        
        # Initialize the accumulated mask with the same size as fg_mask
        if accumulated_mask is None:
            accumulated_mask = np.zeros_like(fg_mask, dtype=np.float32)
        
        # Accumulate the foreground masks
        accumulated_mask += fg_mask.astype(np.float32)
        frame_count += 1

    cap.release()

    # Normalize the accumulated mask
    accumulated_mask /= frame_count

    # Convert accumulated mask to 8-bit image
    accum_mask_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply adaptive thresholding (Binary Threshold method) to find significant changes
    _, thresholded_image = cv2.threshold(accum_mask_uint8, 0, 255, cv2.THRESH_BINARY)

    # Return the thresholded image highlighting significant changes
    return thresholded_image

output_image = threshold_video_movement('test-clogged.mjpeg')
#output_image = detect_clogged_nozzles('test2_0920.mjpeg')
cv2.imwrite('clogged_nozzles_output.png', output_image)
