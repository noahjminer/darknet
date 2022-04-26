from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from darknet_images import depth_detection_on_frame, image_detection_list_on_frame
from queue import Queue
import signal
import math


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, image_queue):
    while global_cap.isOpened():
        ret, frame = global_cap.read()
        if not ret:
            image_queue.put(None)
            frame_queue.put(None)
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.put(frame)
        image_queue.put(frame_rgb)
    # cap.release()


#########################################################################
#
#                   Code Changes Start Here
#
#########################################################################

# Compares pixel differences from a previous frame to current frame.
# Unused in final pipeline
def compare_slices(pre_frame, cur_frame, dim, diff_thresh, thread_result,
                   thread_id):  # diff_thresh is the different threshhold, value from 0 to 1, 0 is no change
    # function return true if the proportion of changes in slice is greater than diff_thresh
    start_thread = time.time()
    x1, x2, y1, y2 = dim
    count = 0
    compress_level = 8  # distance between checking pixels
    print(x1, x2, y1, y2)
    for x in range(x1, x2, compress_level):
        for y in range(y1, y2, compress_level):
            r1, g1, b1 = pre_frame[y, x]
            r2, g2, b2 = cur_frame[y, x]

            color_diff = math.sqrt((int(r1) - int(r2)) ** 2 +
                                   (int(g1) - int(g2)) ** 2 +
                                   (int(b1) - int(b2)) ** 2)

            if color_diff > 5:  # change this to a RBG compare function
                count += compress_level ** 2
                # if pixels are different, assuming that whole section is different

    end_thread = time.time()
    print(count / ((x2 - x1) * (y2 - y1)), '----------', diff_thresh, " in time ", end_thread - start_thread)

    if count / ((x2 - x1) * (y2 - y1)) > diff_thresh:
        thread_result[thread_id] = True
    else:
        thread_result[thread_id] = False
    return


# used to store objects from previous slices in the case there were objects but no frame differences.
def check_object_in_prev_slice(bbox, dim):
    x, y, _, _, = bbox
    x1, x2, y1, y2 = dim
    if (x1 <= x) and (x <= x2) and (y1 <= y) and (y <= y2):
        return True
    return False


# Runs threads to compare each slice with previous
def run_compare_thread(dims, prev_frame, frame, prev_detection):
    thread_tracker = []
    thread_result = [False] * len(dims)
    start_time = time.time()
    for i in range(len(dims)):
        thread_tracker.append(Thread(target=compare_slices, args=(prev_frame, frame, dims[i], 0.05, thread_result, i)))
        thread_tracker[i].start()
    for i in range(len(dims)):
        thread_tracker[i].join()
    print("end in -----------------", time.time() - start_time)
    remain_detection = []
    new_dims = []
    for i in range(len(dims)):
        if thread_result[i] is False:
            dim = dims[i]
            for detection in prev_detection:
                # detection is label, confidence, bbox
                # x, y, w, h = bbox
                if check_object_in_prev_slice(detection[2], dim):
                    remain_detection.append(detection)
        else:
            new_dims.append(dims[i])
    return remain_detection, new_dims


# function for inference thread. Used to run detection methods.
def inference(method, image_queue, detections_queue, fps_queue, dims, network, class_names, class_colors, detection_thresh):
    global prev_frame, prev_detection, global_cap

    while global_cap.isOpened():
        frame = image_queue.get()
        if frame is None:
            break
        prev_time = time.time()

        remain_detection = []
        cur_dims = dims

        prev_frame = frame

        slice_side_length = 0
        if len(dims):
            slice_side_length = dims[0][1] - dims[0][0]
        if method == 'depth':
            detections = depth_detection_on_frame(frame, cur_dims, network, class_names, class_colors, slice_side_length,
                                              detection_thresh)
        else:
            detections = image_detection_list_on_frame(frame, dims, network, class_names, class_colors, detection_thresh)
        for detection in remain_detection:
            detections.append(detection)

        detections_queue.put(detections)

        prev_detection = detections

        fps = float(1 / (time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, True)
    # cap.release()


# function for drawing. Used to render final video. Also writes detection text file.
def drawing(frame_queue, detections_queue, dims, fps_queue, video_width, video_height, class_colors,
            output_filename='result.avi'):
    random.seed(3)  # deterministic bbox colors
    global video, txt_output, frame_count
    video = set_saved_video(global_cap, output_filename, (video_width, video_height))
    while global_cap.isOpened():
        frame = frame_queue.get()
        if frame is None:
            break
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame is not None:
            image = darknet.draw_boxes(detections, frame, class_colors)
            image = darknet.draw_slices(dims, image, class_colors)
            if output_filename is not None:
                video.write(image)
                txt_output.write(str(frame_count) + '\n')
                for label, confidence, bbox in detections:
                    x, y, w, h = bbox
                    txt_output.write("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})\n".format(
                        label, confidence, x, y, w, h))
                frame_count += 1
            if cv2.waitKey(int(fps)) == 27:
                break
    global_cap.release()
    video.release()
    # cv2.destroyAllWindows()


# globals for running concurrently
global_cap = None
darknet_width = 0
darknet_height = 0
video = None

prev_detection = []
prev_frame = None

frame_queue = None
image_queue = None
frame_count = 1

txt_output = None


# runs threads for detecting on and drawing video
def detect_video(args, video_path, dims, detection_thresh, output_filename='result.avi'):
    global global_cap, darknet_width, darknet_height, frame_queue, image_queue, txt_output

    txt_output = open(args.image_path.split('.', 1)[0] + '_detections.txt', 'w')

    frame_queue = Queue()
    image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    # Create network
    if args.method == 'baseline':
        network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    else:
        network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=len(dims)
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    # create threads
    input_path = str2int(video_path)
    global_cap = cv2.VideoCapture(input_path)
    video_width = int(global_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(global_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture_thread = Thread(target=video_capture, args=(frame_queue, image_queue))
    inference_thread = Thread(target=inference, args=(args.method,
        image_queue, detections_queue, fps_queue, dims, network, class_names, class_colors, detection_thresh))
    drawing_thread = Thread(target=drawing, args=(
        frame_queue, detections_queue, dims, fps_queue, video_width, video_height, class_colors, output_filename))

    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()

    capture_thread.join()
    inference_thread.join()
    drawing_thread.join()
    print("Done with video")


def handler(signum, frame):
    global global_cap, video, frame_queue, image_queue
    print("exiting xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if frame_queue:
        frame_queue.put(None)
    if image_queue:
        image_queue.put(None)
    if global_cap:
        global_cap.release()
    if video:
        video.release()
    time.sleep(1)
    print("exiting sucessfully x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x00x0x0x00x0xx0")

# END CUSTOM CODE
signal.signal(signal.SIGINT, handler)