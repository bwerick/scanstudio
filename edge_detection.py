import cv2
import os
import numpy as np
import flor

# Global variables to store points
points = []


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to capture two points for cropping.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        points.append((x, y))
        print(f"Point {len(points)} selected: {x}, {y}")


def crop_with_two_clicks(image_path):
    """
    Allow the user to select a region of interest (ROI) with two clicks.
    :param image_path: Path to the input image.
    :return: Selected points for cropping.
    """
    global points
    points = []  # Reset points
    image = cv2.imread(image_path)
    clone = image.copy()

    cv2.imshow("Original Image", image)
    cv2.setMouseCallback("Original Image", mouse_callback)

    print("Click on the top-left corner and then the bottom-right corner of the ROI.")
    while len(points) < 2:
        cv2.imshow("Original Image", image)
        cv2.waitKey(1)

    # Draw markers on the selected points
    for point in points:
        cv2.circle(clone, point, 5, (0, 0, 255), -1)  # Red circle for markers
    cv2.imshow("Selected Points", clone)
    cv2.waitKey(0)  # Wait for user to confirm
    return points


def track_and_crop_frames(directory, points):
    """
    Track the selected points across frames using optical flow and crop dynamically.
    :param directory: Path to the directory containing frames.
    :param points: Selected points for cropping (from the first frame).
    """
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Load all frames
    frame_paths = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    first_frame = cv2.imread(frame_paths[0])
    h, w = first_frame.shape[:2]

    # Initialize optical flow tracking
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    output_dir = os.path.join(directory, "cropped_frames")
    os.makedirs(output_dir, exist_ok=True)

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track points using optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None, winSize=(21, 21), maxLevel=3
        )

        # Update bounding box based on tracked points
        if status[0][0] and status[1][0]:  # Ensure both points are tracked successfully
            x1, y1 = next_points[0].ravel()
            x2, y2 = next_points[1].ravel()
            x_start, x_end = int(min(x1, x2)), int(max(x1, x2))
            y_start, y_end = int(min(y1, y2)), int(max(y1, y2))

            # Crop the frame
            cropped = frame[
                max(0, y_start) : min(h, y_end), max(0, x_start) : min(w, x_end)
            ]

            # Save the cropped frame
            output_path = os.path.join(output_dir, f"{i:04d}.jpg")
            cv2.imwrite(output_path, cropped)
            print(f"Cropped frame saved to: {output_path}")

            # Update previous frame and points
            prev_gray = gray
            prev_points = next_points
        else:
            print(f"Tracking failed for frame {i}. Reinitializing points.")
            prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=2, qualityLevel=0.01, minDistance=10, mask=None
            )
            if prev_points is None or len(prev_points) < 2:
                print(f"Reinitialization failed for frame {i}. Skipping.")
                continue


if __name__ == "__main__":
    input_dir = flor.arg("input_dir", os.path.join("test_frames", "WebOfBelief"))
    first_frame_path = os.path.join(
        input_dir,
        sorted(
            [
                each
                for each in os.listdir(input_dir)
                if each.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )[0],
    )

    # Step 1: Select points on the first frame
    selected_points = crop_with_two_clicks(first_frame_path)

    # Step 2: Track points and crop frames dynamically
    track_and_crop_frames(input_dir, selected_points)

    print("Cropping completed for all frames.")
    cv2.destroyAllWindows()
