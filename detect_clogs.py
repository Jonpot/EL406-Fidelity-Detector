import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector as apriltag

def threshold_video_movement(video_path: str) -> list[dict]:
    """
    Detect significant changes in a video using background subtraction and thresholding.

    Args:
        video_path: Path to the video file.

    Returns:
        A list of dictionaries, each containing:
        - 'thresholded_image': the thresholded image for the cycle
        - 'fiducial_coordinate': the coordinate of the fiducial (if any)
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
    cycles = []
    current_cycle = None
    state = 'waiting_for_cycle'
    fiducial_coordinate = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if fiducial_coordinate is None:
            fiducial_coordinate = detect_fiducial(frame)

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        movement_amount = np.count_nonzero(fg_mask)
        staleness = 0

        # Define movement thresholds
        low_movement_threshold = 0.025 * fg_mask.size
        high_movement_threshold = 0.05 * fg_mask.size

        if state == 'waiting_for_cycle':
            if movement_amount < low_movement_threshold:
                # Start a new cycle
                current_cycle = {'accum_mask': np.zeros_like(fg_mask, dtype=np.float32),
                                 'frame_count': 0}
                state = 'in_cycle'
                print("Starting new cycle")
        elif state == 'in_cycle':
            # Accumulate the fg_mask
            current_cycle['accum_mask'] += fg_mask.astype(np.float32)
            current_cycle['frame_count'] += 1

            if movement_amount > high_movement_threshold:
                # End of cycle
                cycles.append(current_cycle)
                current_cycle = None
                state = 'waiting_for_movement'
                print("End of cycle")
        elif state == 'waiting_for_movement':
            if movement_amount < low_movement_threshold:
                state = 'between_cycles'
                print("Between cycles")
        elif state == 'between_cycles':
            if movement_amount > high_movement_threshold:
                # Robot moving back into frame
                print(f'Staleness: {staleness}')
                staleness = 0
                state = 'waiting_for_cycle'
                print("Waiting for cycle")
            else:
                staleness += 1
                if staleness > 50:
                    # 50 frames of no movement, the robot isn't coming back for a second cycle at this point
                    # This might need to be adjusted
                    break

    # After processing all frames
    if current_cycle is not None:
        cycles.append(current_cycle)

    cap.release()

    # For each cycle, normalize the accumulated mask and process
    for cycle in cycles:
        frame_count = cycle['frame_count']
        accumulated_mask = cycle['accum_mask']
        accumulated_mask /= frame_count

        # Convert accumulated mask to 8-bit image
        accum_mask_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(accum_mask_uint8, 0, 255, cv2.THRESH_BINARY)

        cycle['thresholded_image'] = thresholded_image
        cycle['fiducial_coordinate'] = fiducial_coordinate

    return cycles


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


def front_homography(fiducial_coordinate: tuple[float, float], image: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to the image using the front fiducial coordinate.

    Args:
        fiducial_coordinate: The coordinate of the front fiducial.
        image: The input image.

    Returns:
        The warped image after homography transformation.
    """
    if fiducial_coordinate is None:
        pts_src = np.array([
            [388, 180],  # Top-left 388, 180
            [850, 285],  # Top-right 850, 285
            [388, 302],  # Bottom-left 388, 302
            [850, 375]  # Bottom-right 850, 375
            # This is for front nozzles!
        ], dtype=float)
        print("No fiducials detected. Using absolute coordinates.\n")
    else:
        x, y = fiducial_coordinate.x, fiducial_coordinate.y

        pts_src = np.array([
            [x - 512, y - 420],  # Top-left 388, 180
            [x - 50, y - 315],  # Top-right 850, 285
            [x - 512, y - 298],  # Bottom-left 388, 302
            [x - 50, y - 225]  # Bottom-right 850, 375
        ], dtype=float)
        print("Fiducials detected. Using relative coordinates.\n")

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


def back_homography(fiducial_coordinate: tuple[float, float], image: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to the image using the back fiducial coordinate.

    Args:
        fiducial_coordinate: The coordinate of the back fiducial.
        image: The input image.

    Returns:
        The warped image after homography transformation.
    """
    # print(len(detections))
    if fiducial_coordinate is None:
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

        x, y = fiducial_coordinate.x, fiducial_coordinate.y

        # Debug: Draw fiducial on the first frame
        # cv2.circle(first_frame, (int(center.x), int(center.y)), 5, (0, 0, 255), -1)
        # for i, (label, point) in enumerate([('Top-left', (center.x - 512, center.y - 420)),
        #                                    ('Top-right', (center.x - 50, center.y - 315)),
        #                                    ('Bottom-left', (center.x - 512, center.y - 298)),
        #                                    ('Bottom-right', (center.x - 50, center.y - 225))]):
        #      cv2.circle(first_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
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

    #plt.figure(figsize=(10, 6))
    #plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    #plt.title("Warped Image with Homography Transformation (One Fiducial)")
    #plt.axis('off')
    #plt.show()

    return warped_image

def classify_nozzles(warped_image: np.ndarray, section: str ='front') -> tuple[dict[str, list[bool]], float]:
    """
    Classify nozzle regions as clogged or not clogged based on white pixel ratio.

    Args:
        warped_image: The input image of nozzle regions after homography.
        section: Specify which section of nozzles to process ('front' or 'back').

    Returns:
        A tuple containing:
        - A dictionary with the nozzle status.
        - The average white_ratio across nozzles.
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
    threshold_z = -1.0  # Z-score threshold for 2 standard deviations

    nozzle_status = []

    # Detect clogged nozzles based on z-score and <5% threshold
    for i, ratio in enumerate(white_ratios):
        z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        is_clogged = z_score < threshold_z or ratio < 0.05
        nozzle_status.append(is_clogged)

    # Return the status of front or back nozzles and the mean white_ratio
    return {section: nozzle_status}, mean_ratio
    