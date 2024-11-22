import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
_ = plt.get_cmap("Reds")
_ = np.zeros(1)

# Device index (4) on my machine with DroidCam
device_index = 4
# Points to track
points_to_track = 5

# For detecting boundaries
contour_range = [20, 2000]
# DEBUGGING:
vary_contour_range = False

# For detecting dark regions
threshold_range = [15, 255]


def vary_range():
    # 3 change filter range
    index = 1
    minval, maxval = 10, 4000
    stepsize = 30
    rise = True
    if rise:
        contour_range[index] += stepsize
    else:
        contour_range[index] -= stepsize
    if contour_range[index] > maxval:
        rise = False
    if contour_range[index] < minval:
        rise = True
    print(f"contour_range: {contour_range}")


def loop():
    while True:
        if vary_contour_range:
            vary_range()

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Show the video stream if desired
        if args.show:
            cv2.imshow("frame", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # apply grey color filter
        gray = frame.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # apply binary color filter to detect dark regions better
        _, thresh = cv2.threshold(gray, threshold_range[0], threshold_range[1],
                                  cv2.THRESH_BINARY_INV)
        cv2.imshow("thresh", thresh)

        # find contours of the binary image
        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # track largest black points
        black_points = []
        global area
        for contour in contours:
            area = cv2.contourArea(contour)
            if contour_range[0] < area < contour_range[1]:
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                black_points.append((cx, cy))
            # draw contours on frame
            # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # draw points on frame
        overlay = frame.copy()
        for i, point in enumerate(black_points):
            print(f"point {i+1}: {point}")
            if i+1 > points_to_track:
                break
            cv2.circle(overlay, point, 8, (0, 0, 255), -1)
            alpha = 0.9  # Transparency factor
            overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.imshow("overlay", overlay)
    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # For parsing arguments
    parser = argparse.ArgumentParser(description="")
    # Add the --show argument
    parser.add_argument("--show", action="store_true", help="Display raw")
    args = parser.parse_args()

    # Initialize the camera
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("Camera opened")

    # Start the loop
    loop()
