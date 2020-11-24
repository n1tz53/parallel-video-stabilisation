"""
Written By: ntnkmar@gmail.com
Following code is heavily inspired from:
- SIMPLE VIDEO STABILIZATION USING OPENCV (http://nghiaho.com/?p=2093)
- https://towardsdatascience.com/faster-video-processing-in-python-using-parallel-computing-25da1ad4a01
- https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

Approach: Instead of stabilising video in sequential manner we divide it into
components based on core available and then track feature for each component
and later apply feature transformation to each component and merge all components
Disadvantage: you can see slight change in angle where video are merged
"""

import os
import time
import multiprocessing as mp
import subprocess as sp
import numpy as np
import cv2 as cv

smoothing_radius = 30
frame_jump = 1
ax_crop = 30
output_file_name = 'stabilised_output.mp4'
path = None


def moving_avg(curve, radius):
    # Define the filter
    conv_filter = np.ones(2 * radius + 1)
    conv_filter = conv_filter / len(conv_filter)
    # Add padding to the boundaries
    curve = np.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve = np.convolve(curve, conv_filter, mode='same')
    # Remove padding
    curve = curve[radius:-radius]
    # return smoothed curve
    return curve


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    smoothed_trajectory[:, 0] = moving_avg(trajectory[:, 0], radius=smoothing_radius)
    smoothed_trajectory[:, 1] = moving_avg(trajectory[:, 1], radius=smoothing_radius)
    smoothed_trajectory[:, 2] = moving_avg(trajectory[:, 2], radius=smoothing_radius)
    return smoothed_trajectory


def track_features(group_number):
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_jump * group_number)
    ok, prev = cap.read()
    prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    # Pre-define transformation-store array
    transforms = []
    # start = time.time()
    frame = 1
    while frame < frame_jump:
        # Read next frame
        ok, curr = cap.read()
        if not ok:
            break
        # Convert to grayscale
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        # Detect feature points in previous frame
        prev_pts = cv.goodFeaturesToTrack(prev_gray,
                                          maxCorners=150,
                                          qualityLevel=0.2,
                                          minDistance=30,
                                          blockSize=3)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # Sanity check
        assert prev_pts.shape == curr_pts.shape
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        # Find transformation matrix
        m = cv.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less
        # Extract translation
        dx, dy = m[0, 2], m[1, 2]
        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])
        # Store transformation
        transforms.append([dx, dy, da])
        # Move to next frame
        prev_gray = curr_gray
        # print(f'frame: {frame + 1}/{num_frames} tracked points: {prev_pts.shape[0]}')
        frame += 1
    # print(f'time to calculate tracking: {time.time() - start}')
    cap.release()
    return group_number, np.array(transforms, dtype=np.float32)


def apply_transformation(group_number, transforms):
    cap = cv.VideoCapture(path)
    # Get width and height of video stream
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Define the codec for output video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv.CAP_PROP_FPS)
    ay_crop = int(ax_crop * (h / w))
    # Set up output video
    out = cv.VideoWriter()
    out.open("output{}.mp4".format(group_number), fourcc, fps, (w, h), True)
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smooth(trajectory) - trajectory
    # Calculate newer transformation array
    smooth_transforms = transforms + difference
    # Reset stream to first frame of group
    cap.set(cv.CAP_PROP_POS_FRAMES, group_number * frame_jump)
    # Write frame_jump transformed frames
    for idx in range(frame_jump - 2):
        # Read next frame
        ok, frame = cap.read()
        if not ok:
            break
        # Extract transformations from the new transformation array
        dx = smooth_transforms[idx, 0]
        dy = smooth_transforms[idx, 1]
        da = smooth_transforms[idx, 2]
        # Reconstruct transformation matrix accordingly to new values
        m = [[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]]
        m = np.array(m, dtype=np.float32)
        # Apply affine wrapping to the given frame
        stabilized_frame = cv.warpAffine(frame, m, (w, h))
        # Fix borders
        stabilized_frame = stabilized_frame[ay_crop:-ay_crop, ax_crop:-ax_crop, :]
        stabilized_frame = cv.resize(stabilized_frame, (w, h))
        out.write(stabilized_frame)
    out.release()
    cap.release()
    return


# call stabilization funtion with path of file and number of cores
def stabilize_video(file_path, num_process):
    global path
    path = file_path
    # Process the video by splitting it into as many fragments as the number of virtual cores you have
    num_processes = min(mp.cpu_count(), num_process)
    global frame_jump
    frame_jump = int(cv.VideoCapture(path).get(cv.CAP_PROP_FRAME_COUNT) // num_processes)
    print(f"CPU count: {mp.cpu_count()} num_processes: {num_processes} frame jump: {frame_jump}")
    end_time = time.time()
    res = mp.Pool(num_processes).map(track_features, range(num_processes))
    print(f"time to track features: {time.time() - end_time}")
    # print(res)
    end_time = time.time()
    mp.Pool(num_processes).starmap(apply_transformation, res)
    print(f"time taken for transformation: {time.time() - end_time}")
    # Create a list of output files and store the file names in a txt file
    list_of_output_files = ["output{}.mp4".format(i) for i in range(num_processes)]
    with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))
    # use ffmpeg to combine the video output files
    cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy "
    ffmpeg_cmd = cmd + output_file_name
    sp.Popen(ffmpeg_cmd, shell=True).wait()
    # Remove the temporary output files
    for f in list_of_output_files:
        os.remove(f)
    os.remove("list_of_output_files.txt")
    return
