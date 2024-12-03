import numpy as np
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
# import glob

# calibration images:
chessboard_size = (7, 7)  # number of corners in the chessboard
chessboard_square_size = 30  # mm, size of each square in the chessboard

# Device index on my machine with DroidCam
device_index = 0
# iPhone 8:
# [[540.69756264   0.         311.16613271]
#  [  0.         529.13613376 239.35617656]
#  [  0.           0.           1.        ]]

# Points to track
max_points_to_track = 4
min_points_for_projection = 4

# Ring parameters
radius = 20  # mm
thickness = 5
model_points = np.array([
    [radius,  0,       thickness],  # Replace with actual 3D coordinates
    [0,       radius,  thickness],
    [-radius, 0,       thickness],
    [0,       -radius, thickness],
], dtype=np.float32)

# For detecting boundaries
contour_range = [100, 2000]
# For detecting dark regions
threshold_range = [20, 255]  # 15, 255


def interact():
    key = cv.waitKey(1) & 0xFF

    # CONTOUR RANGE
    if key == ord('w'):
        contour_range[1] += 10
        print(f"contour_range upper boundary increased: {contour_range}")
    elif key == ord('s'):
        contour_range[1] -= 10
        print(f"contour_range upper boundary decreased: {contour_range}")
    elif key == ord('d'):
        contour_range[0] += 10
        print(f"contour_range lower boundary increased: {contour_range}")
    elif key == ord('a'):
        contour_range[0] -= 10
        print(f"contour_range lower boundary decreased: {contour_range}")

    # THRESHOLD RANGE
    elif key == ord('i'):
        threshold_range[1] += 10
        print(f"threshold_range upper boundary increased: {threshold_range}")
    elif key == ord('k'):
        threshold_range[1] -= 10
        print(f"threshold_range upper boundary decreased: {threshold_range}")
    elif key == ord('l'):
        threshold_range[0] += 10
        print(f"threshold_range lower boundary increased: {threshold_range}")
    elif key == ord('j'):
        threshold_range[0] -= 10
        print(f"threshold_range lower boundary decreased: {threshold_range}")

    # Exit the loop when 'q' is pressed
    elif key == ord('q'):
        return True

    contour_range[0] = max(contour_range[0], 1)
    contour_range[1] = max(contour_range[1], 1)
    threshold_range[0] = max(threshold_range[0], 1)
    threshold_range[1] = max(threshold_range[1], 1)

    return False


def loop():
    print("Press Enter to start projecting 3D points")
    start_projection = False

    while True:
        # for imputs
        if interact():
            break

        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Show the video stream if desired
        if args.show:
            cv.imshow("frame", frame)

        # DEBUG, USE ONLY RED CHANNEL FOR TEST
        red = frame[:, :, 2]
        _, thresh_red = cv.threshold(red, threshold_range[0],
                                     threshold_range[1],
                                     cv.THRESH_BINARY)
        cv.imshow("RED", thresh_red)

        # apply grey color filter
        gray = frame.copy()
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        # apply binary color filter to detect dark regions better
        _, thresh = cv.threshold(gray, threshold_range[0],
                                 threshold_range[1], cv.THRESH_BINARY_INV)
        cv.imshow("Threshold", thresh)

        # find contours of the binary image
        contours, _ = cv.findContours(thresh,
                                      cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # track largest black points
        black_points = np.empty((0, 2), dtype=np.int32)
        global area
        for contour in contours:
            area = cv.contourArea(contour)
            if contour_range[0] < area < contour_range[1]:
                M = cv.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                black_points = np.append(black_points, [[cx, cy]], axis=0)
            # draw contours on frame
            # cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # draw points on frame
        overlay = frame.copy()
        for i, point in enumerate(black_points):
            # print(f"point {i+1}: {point}")
            if i+1 > max_points_to_track:
                break
            cv.circle(overlay, tuple(point), 8, (0, 0, 255), -1)
            alpha = 0.9  # Transparency factor
            overlay = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv.imshow("Overlay", overlay)

        key = cv.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            start_projection = True
        if start_projection and len(black_points) is min_points_for_projection:
            black_points = black_points.astype(np.float32)
            if black_points.shape[0] > 0:
                reprojected_points = project3d(black_points)

                # draw reprojected points on frame
                reprojected_points.astype(np.int32)
                reprojection = frame.copy()
                if reprojected_points is not None:
                    for i, point in enumerate(reprojected_points):
                        cv.circle(reprojection, tuple(point[0].astype(int)),
                                  8, (0, 0, 255), -1)
                        alpha = 0.9  # Transparency factor
                        reprojection = cv.addWeighted(overlay, alpha, frame,
                                                      1 - alpha, 0)
                    cv.imshow("Reprojection", reprojection)

        # Exit the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv.destroyAllWindows()


def project3d(image_points):
    image_points = image_points.reshape(-1, 1, 2)
    success, rvec, tvec = cv.solvePnP(
        model_points,    # 3D points in model space
        image_points,    # Corresponding 2D points in the image
        mtx,   # Camera intrinsic matrix
        dist,     # Distortion coefficients
        flags=cv.SOLVEPNP_P3P)
    # flags=cv.SOLVEPNP_ITERATIVE,
    # useExtrinsicGuess=True)

    if not success:
        print("Failed to solve PnP")
        return
    R, _ = cv.Rodrigues(rvec)
    camera_coordinates = np.dot(R, model_points.T).T + tvec.T
    # TODO: define wold coordinate system and project 3d points to it

    # Update the plot
    ax.clear()
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, 50])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=20., azim=30)  # Set common viewing angle
    ax.scatter(camera_coordinates[:, 0], camera_coordinates[:, 1],
               camera_coordinates[:, 2], c='r', marker='o')
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

    reprojected_points, _ = cv.projectPoints(model_points, rvec, tvec,
                                             mtx, dist)
    return reprojected_points


def calibrate():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= chessboard_square_size  # Scale by the size of the squares

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press Enter to take a picture")
    successes = 0
    while successes < 10:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv.imshow('Calibration Feed', frame)
        key = cv.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboard_size,
                                                    None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                # cv.drawChessboardCorners(frame, chessboard_size,corners2,ret)
                # cv.imshow('Success!', frame)
                successes += 1
                print(f"Picture {successes}/10 taken and analyzed")
                # time.sleep(500)
            else:
                print("Chessboard corners not found")

        elif key == ord('q'):  # Quit when 'q' is pressed
            break

    cap.release()
    cv.destroyAllWindows()

    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                          gray.shape[::-1],
                                                          None, None)
        if not ret:
            print("Calibration failed")
            return
        print(f"Camera matrix: {mtx}")
        print(f"Distortion coefficients: {dist}")
        # print(f"Rotation vectors: {rvecs}")
        # print(f"Translation vectors: {tvecs}")
        np.savez("calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs,
                 tvecs=tvecs)
    else:
        print("No calibration data collected")


if __name__ == "__main__":
    # For parsing arguments
    parser = argparse.ArgumentParser(description="")
    # Add the --show argument
    parser.add_argument("--show", action="store_true", help="Display raw")
    args = parser.parse_args()

    # Initialize the camera
    cap = cv.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("Camera opened")

    # Create and move windows
    cv.namedWindow("Overlay")
    cv.namedWindow("Threshold")
    cv.namedWindow("Reprojection")
    cv.moveWindow("Overlay", 0, 0)
    cv.moveWindow("Threshold", 650, 0)
    cv.moveWindow("Reprojection", 0, 670)

    # Ask for recalibration
    print("Recalibrate setup? Press Enter for Yes, 'n' for No")
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('y'):
            print("Recalibrating setup...")
            calibrate()
            break
        elif key == ord('n'):
            print("Skipping recalibration...")
            break

    calibration_data = np.load("calibration.npz")
    mtx = calibration_data["mtx"]
    dist = calibration_data["dist"]
    print(f"Camera matrix: \n {mtx}")
    print(f"Distortion coefficients: \n {dist}")

    # prepare plot:
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=20., azim=30)  # Set common viewing angle
    plt.get_current_fig_manager().window.wm_geometry("+670+650")

    # Start the loop
    loop()
