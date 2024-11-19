import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector as apriltag
import matplotlib.pyplot as plt

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
    fiducial_coordinates = None
    i = 0
    staleness = 0
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        #if i % 3 == 0:
        #    continue # Skip every other frame to speed up processing
        if fiducial_coordinates is None:
            fiducial_coordinates = detect_fiducials(frame)

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        movement_amount = np.count_nonzero(fg_mask)

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
            if movement_amount > high_movement_threshold and staleness > 10:
                # Robot moving back into frame
                print(f'Staleness: {staleness}')
                staleness = 0
                state = 'waiting_for_cycle'
                print("Waiting for cycle")
            else:
                staleness += 1
                if staleness > 50:
                    print("Staleness limit reached")
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
        cycle['fiducial_coordinates'] = fiducial_coordinates

    return cycles


def detect_fiducials(frame: np.ndarray) -> dict[int, np.ndarray]:
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

    centers = {}
    for detection in detections:
        centers[detection.getId()] = (detection.getCenter())

    return centers


def front_homography(fiducial_coordinates: dict[int, tuple[float, float]], image: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to the image using the front fiducial coordinate.

    Args:
        fiducial_coordinate: The coordinate of the front fiducial.
        image: The input image.

    Returns:
        The warped image after homography transformation.
    """
    absolute_coordinates = np.array([
            [449, 247],
            [902, 321],
            [455, 421],
            [914, 467] 
            # This is for front nozzles!
        ], dtype=float)
    if fiducial_coordinates is None:
        pts_src = absolute_coordinates
        print("No fiducials detected. Using absolute coordinates.\n")
    elif len(fiducial_coordinates) == 1:
        if 2 in fiducial_coordinates:
            x, y = fiducial_coordinates[2].x, fiducial_coordinates[2].y
            x0, y0 = (995, 672)
            # pts_src is thus the absolute coordinates of the fiducial plus the difference between the absolute coordinates of the fiducial and the relative coordinates of the nozzles
            pts_src = absolute_coordinates + np.array([[x - x0, y - y0]*4], dtype=float) 
        else:
            x, y = fiducial_coordinates[1].x, fiducial_coordinates[1].y
            x0, y0 = (277, 184)
            pts_src = absolute_coordinates + np.array([[x - x0, y - y0]*4], dtype=float)
        print("One Fiducial detected. Using relative coordinates.\n")
    else:
        print("Multiple fiducials detected. Using fiducial coordinates.")
        x1, y1 = fiducial_coordinates[1].x, fiducial_coordinates[1].y
        x10, y10 = (277, 184)
        x2, y2 = fiducial_coordinates[2].x, fiducial_coordinates[2].y
        x20, y20 = (995, 672)
        scale_x = (x2-x1)/(x20-x10)
        scale_y = (y2-y1)/(y20-y10)
        pts_src = (absolute_coordinates - np.array([[x10, y10],[x10, y10],[x10, y10],[x10, y10]], dtype=float)) * np.array([[scale_x, scale_y], [scale_x, scale_y], [scale_x, scale_y], [scale_x, scale_y]], dtype=float)
        pts_src = pts_src + np.array([[x1, y1], [x1, y1], [x1, y1], [x1, y1]], dtype=float)

    width = 400
    height = 300
    pts_dst = np.array([
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [0, height - 1],  # Bottom-left
        [width - 1, height - 1]  # Bottom-right
    ], dtype=float)

    # Calculate the homography matrix
    print(fiducial_coordinates)
    print(pts_src)
    print(pts_dst)
    homography_matrix, status = cv2.findHomography(pts_src, pts_dst)

    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

    # Display the warped image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image with Homography Transformation (One Fiducial)")
    plt.axis('off')
    plt.show()

    return warped_image


def back_homography(fiducial_coordinates: dict[int,tuple[float, float]], image: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to the image using the back fiducial coordinate.

    Args:
        fiducial_coordinate: The coordinate of the back fiducial.
        image: The input image.

    Returns:
        The warped image after homography transformation.
    """
    # print(len(detections))
    absolute_coordinates = np.array([
                            [380, 239], 
                            [844, 334], 
                            [388, 428], 
                            [846, 459] 
                            # This is for BACK nozzles!
                        ], dtype=float)
    if fiducial_coordinates is None:
        pts_src = absolute_coordinates
        print("No fiducials detected. Using absolute coordinates.\n")
    elif len(fiducial_coordinates) == 1:
        if 2 in fiducial_coordinates:
            x, y = fiducial_coordinates[2].x, fiducial_coordinates[2].y
            x0, y0 = (995, 672)
            # pts_src is thus the absolute coordinates of the fiducial plus the difference between the absolute coordinates of the fiducial and the relative coordinates of the nozzles
            pts_src = absolute_coordinates + np.array([[x - x0, y - y0]*4], dtype=float) 
        else:
            x, y = fiducial_coordinates[1].x, fiducial_coordinates[1].y
            x0, y0 = (277, 184)
            pts_src = absolute_coordinates + np.array([[x - x0, y - y0]*4], dtype=float)
        print("One Fiducial detected. Using relative coordinates.\n")
    else:
        print("Multiple fiducials detected. Using fiducial coordinates.")
        x1, y1 = fiducial_coordinates[1].x, fiducial_coordinates[1].y
        x10, y10 = (277, 184)
        x2, y2 = fiducial_coordinates[2].x, fiducial_coordinates[2].y
        x20, y20 = (995, 672)
        scale_x = (x2-x1)/(x20-x10)
        scale_y = (y2-y1)/(y20-y10)
        pts_src = (absolute_coordinates - np.array([[x10, y10],[x10, y10],[x10, y10],[x10, y10]], dtype=float)) * np.array([[scale_x, scale_y], [scale_x, scale_y], [scale_x, scale_y], [scale_x, scale_y]], dtype=float)
        pts_src = pts_src + np.array([[x1, y1], [x1, y1], [x1, y1], [x1, y1]], dtype=float)

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

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image with Homography Transformation (One Fiducial)")
    plt.axis('off')
    plt.show()

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
    threshold_z = -1.75  # Z-score threshold for 2 standard deviations

    nozzle_status = []

    # Detect clogged nozzles based on z-score and <5% threshold
    for i, ratio in enumerate(white_ratios):
        z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        is_clogged = z_score < threshold_z or ratio < 0.05
        print(f"Nozzle {i + 1}: White ratio = {ratio:.2f}, Z-score = {z_score:.2f} (threshold = {threshold_z}), Clogged = {is_clogged}")
        nozzle_status.append(is_clogged)

    # Return the status of front or back nozzles and the mean white_ratio
    return {section: nozzle_status}, mean_ratio
    