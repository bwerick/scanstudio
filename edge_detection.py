import cv2
import os
import numpy as np
import flordb as flor

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
    assert image is not None, "Failed to load image."
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
    assert first_frame is not None, "Failed to load the first frame."
    h, w = first_frame.shape[:2]

    # Initialize optical flow tracking
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)

    output_dir = os.path.join(directory, "cropped_frames")
    os.makedirs(output_dir, exist_ok=True)

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        assert frame is not None, f"Failed to load frame: {frame_path}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Track points using optical flow (preallocate nextPts to satisfy type stubs)
        next_pts_init = np.empty_like(prev_points)
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            next_pts_init,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Update bounding box based on tracked points
        if next_points is not None and status is not None and bool(np.all(status.ravel())):
            pts = next_points.reshape(-1, 2)
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x_start = int(max(0, min(x1, x2)))
            x_end   = int(min(w, max(x1, x2)))
            y_start = int(max(0, min(y1, y2)))
            y_end   = int(min(h, max(y1, y2)))

            if x_end > x_start and y_end > y_start:
                cropped = frame[y_start:y_end, x_start:x_end]
                output_path = os.path.join(output_dir, f"{i:04d}.jpg")
                cv2.imwrite(output_path, cropped)
                print(f"Cropped frame saved to: {output_path}")

            prev_gray = gray
            prev_points = next_points.reshape(-1, 1, 2)
        else:
            print(f"Tracking failed for frame {i}. Reinitializing points.")
            # Optional: restrict detection near last ROI
            pad = 20
            xs, xe = max(0, int(min(x1, x2)) - pad), min(w, int(max(x1, x2)) + pad)
            ys, ye = max(0, int(min(y1, y2)) - pad), min(h, int(max(y1, y2)) + pad)
            mask = np.zeros_like(gray)
            cv2.rectangle(mask, (xs, ys), (xe, ye), 255, -1)

            prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=2, qualityLevel=0.01, minDistance=10, mask=mask
            )
            if prev_points is None or len(prev_points) < 2:
                print(f"Reinitialization failed for frame {i}. Skipping.")
                continue
            prev_points = prev_points.astype(np.float32).reshape(-1, 1, 2)
            prev_gray = gray


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
