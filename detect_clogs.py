import cv2
import numpy as np

<<<<<<< Updated upstream
def threshold_video_movement(video_path: str) -> np.ndarray:
=======

def threshold_video_movement(video_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
    current_row = 0
    between_nozzles = False

    fiducial_coordinate = None
>>>>>>> Stashed changes

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        # if more than 20% of the frame is moving, this is a significant change, and we break and don't accumulate
<<<<<<< Updated upstream
        #print(np.count_nonzero(fg_mask))
        if frame_count != 0 and np.count_nonzero(fg_mask) > 0.1 * fg_mask.size:
            break
        
        # Initialize the accumulated mask with the same size as fg_mask
        if accumulated_mask is None:
            accumulated_mask = np.zeros_like(fg_mask, dtype=np.float32)
        
        # Accumulate the foreground masks
        accumulated_mask += fg_mask.astype(np.float32)
=======
        # print(np.count_nonzero(fg_mask))
        # print(np.count_nonzero(fg_mask), 0.1 * fg_mask.size)
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

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
output_image = threshold_video_movement('test-clogged.mjpeg')
#output_image = detect_clogged_nozzles('test2_0920.mjpeg')
cv2.imwrite('clogged_nozzles_output.png', output_image)
=======

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


def front_homography(fiducial_coordinate, image):
    detections = fiducial_coordinate
    if detections is None:
        pts_src = np.array([
            [388, 180],  # Top-left 388, 180
            [850, 285],  # Top-right 850, 285
            [388, 302],  # Bottom-left 388, 302
            [850, 375]  # Bottom-right 850, 375
            # This is for front nozzles!
        ], dtype=float)
        print("No fiducials detected. Using absolute coordinates.\n")
    else:
        fiducial_coord = detections
        x, y = fiducial_coord.x, fiducial_coord.y

        pts_src = np.array([
            [x - 512, y - 420],  # Top-left 388, 180
            [x - 50, y - 315],  # Top-right 850, 285
            [x - 512, y - 298],  # Bottom-left 388, 302
            [x - 50, y - 225]  # Bottom-right 850, 375
        ], dtype=float)
        print("Fiducials detected. Using relative coordinates.\n")

    # The following code is to load directly the processed front image, uncomment to use
    # image = cv2.imread(image_path)
    # if image is None:
    #    raise FileNotFoundError(f"Could not load the image at path: {image_path}")

    width = 400
    height = 300
    pts_dst = np.array([
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [0, height - 1],  # Bottom-left
        [width - 1, height - 1]  # Bottom-right
    ], dtype=float)

    # Calculate the homography matrix
    homography_matrix, status = cv2.findHomography(pts_src, pts_dst)

    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

    # Display the warped image
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    # plt.title("Warped Image with Homography Transformation (One Fiducial)")
    # plt.axis('off')
    # plt.show()

    return warped_image


def back_homography(fiducial_coordinate, image):
    detections = fiducial_coordinate

    if detections is None:
        pts_src = np.array([
            [422, 180],  # Top-left 388, 180
            [790, 285],  # Top-right 850, 285
            [422, 302],  # Bottom-left 388, 302
            [790, 375]  # Bottom-right 850, 375
            # This is for BACK nozzles!
        ], dtype=float)
        print("No fiducials detected. Using absolute coordinates.\n")
    else:
        # There should be 1 fiducials:
        # Assuming only one fiducial is used for homography calculation

        fiducial_coord = detections
        x, y = fiducial_coord.x, fiducial_coord.y

        # Debug: Draw fiducial on the first frame
        # cv2.circle(first_frame, (int(center.x), int(center.y)), 5, (0, 0, 255), -1)
        # for i, (label, point) in enumerate([('Top-left', (center.x - 512, center.y - 420)),
        #                                    ('Top-right', (center.x - 50, center.y - 315)),
        #                                    ('Bottom-left', (center.x - 512, center.y - 298)),
        #                                    ('Bottom-right', (center.x - 50, center.y - 225))]):
        #    cv2.circle(first_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        # plt.figure(figsize=(10, 6))
        # plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        # plt.title("First Frame with Fiducials Marked")
        # plt.axis('off')
        # plt.show()

        pts_src = np.array([  # 900, 600
            [x - 578, y - 420],  # Top-left 422, 180
            [x - 110, y - 315],  # Top-right 790, 285
            [x - 578, y - 298],  # Bottom-left 422, 302
            [x - 110, y - 225]  # Bottom-right 790, 375
        ], dtype=float)
        print("Fiducials detected. Using relative coordinates.\n")

    # Used to load the image (back/front) directly
    # image = cv2.imread(image_path)
    # if image is None:
    #    raise FileNotFoundError(f"Could not load the image at path: {image_path}")

    width = 400
    height = 300
    pts_dst = np.array([
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [0, height - 1],  # Bottom-left
        [width - 1, height - 1]  # Bottom-right
    ], dtype=float)

    # Calculate the homography matrix
    homography_matrix, status = cv2.findHomography(pts_src, pts_dst)

    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

    # The following code is to draw homography transformation results, uncomment to use
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    # plt.title("Warped Image with Homography Transformation (One Fiducial)")
    # plt.axis('off')
    # plt.show()

    return warped_image


def classify_nozzles(warped_image, section='front'):
    """
    Classify nozzle regions as clogged or not clogged based on white pixel ratio.

    Args:
        warped_image: The input image of nozzle regions after homography.
        section: Specify which section of nozzles to process ('front' or 'back').

    Returns:
        A dictionary containing the clogging status of the nozzles.
    """
    num_nozzles = 16
    width = warped_image.shape[1]
    height = warped_image.shape[0]

    nozzle_width = width // num_nozzles
    white_ratios = []

    for i in range(num_nozzles):
        x_start = i * nozzle_width
        x_end = (i + 1) * nozzle_width if (i + 1) < num_nozzles else width

        y_start = int(0.10 * height)
        y_end = int(0.90 * height)
        nozzle_region = warped_image[y_start:y_end, x_start:x_end]

        if len(nozzle_region.shape) == 3:
            nozzle_region = cv2.cvtColor(nozzle_region, cv2.COLOR_BGR2GRAY)

        white_pixels = cv2.countNonZero(nozzle_region)
        total_pixels = nozzle_region.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        white_ratios.append(white_ratio)

    # Calculate mean and standard deviation for statistical outlier detection
    mean_ratio = np.mean(white_ratios)
    std_ratio = np.std(white_ratios)
    threshold_z = -2.0  # Z-score threshold for 2 standard deviations

    nozzle_status = []

    # Detect clogged nozzles based on z-score and <5% threshold
    for i, ratio in enumerate(white_ratios):
        z_score = (ratio - mean_ratio) / std_ratio
        is_clogged = z_score < threshold_z or ratio < 0.05  # less than 5% white pixels or 2 std deviations behind
        nozzle_status.append(is_clogged)

    # The following code is to Draw bounding boxes and add text on the warped image, uncomment to use
    # if len(warped_image.shape) == 2:
    #    warped_image_color = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
    # else:
    #    warped_image_color = warped_image

    # for i in range(num_nozzles):
    #    x_start = i * nozzle_width
    #    x_end = (i + 1) * nozzle_width if (i + 1) < num_nozzles else width
    #    y_start = int(0.10 * height)
    #    y_end = int(0.90 * height)

    #    cv2.rectangle(warped_image_color, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    #    text_value = int(white_ratios[i] * 100)
    #    text_color = (0, 0, 204) if white_ratios[i] <= 0.15 else (255, 0, 0)
    #    text_position = (x_start + 2, y_start - 5)
    #    cv2.putText(warped_image_color, f"{text_value}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Display the warped image with nozzle regions marked
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(warped_image_color, cv2.COLOR_BGR2RGB))
    # plt.title(f"Warped Image with Nozzle Regions Marked ({section.capitalize()} Nozzles)")
    # plt.axis('off')
    # plt.show()

    # Return the status of front or back nozzles
    return {section: nozzle_status}
>>>>>>> Stashed changes
