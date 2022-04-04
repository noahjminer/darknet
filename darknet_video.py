from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from darknet_images import depth_detection_on_frame
from queue import Queue


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="result.avi",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise (ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


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


def inference(image_queue, detections_queue, fps_queue, dims, network, class_names, class_colors, detection_thresh):
    while global_cap.isOpened():
        frame = image_queue.get()
        if frame is None:
            break
        prev_time = time.time()
        detections = depth_detection_on_frame(frame, dims, network, class_names, class_colors, detection_thresh)
        detections_queue.put(detections)
        fps = int(1 / (time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, True)
    # cap.release()


def drawing(frame_queue, detections_queue, fps_queue, video_width, video_height, class_colors,
            output_filename='result.avi'):
    random.seed(3)  # deterministic bbox colors

    video = set_saved_video(global_cap, output_filename, (video_width, video_height))
    while global_cap.isOpened():
        frame = frame_queue.get()
        if frame is None:
            break;
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            # for label, confidence, bbox in detections:
            #    bbox_adjusted = convert2original(frame, bbox)
            #    detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections, frame, class_colors)
            # if not args.dont_show:
            #     cv2.imshow('Inference', image)
            if output_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    global_cap.release()
    video.release()
    # cv2.destroyAllWindows()


global_cap = None
darknet_width = 0
darknet_height = 0


def detect_video(args, video_path, dims, detection_thresh, output_filename='result.avi'):
    global global_cap, darknet_width, darknet_height

    frame_queue = Queue()
    image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=len(dims)
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    # figure out webcam as well
    input_path = str2int(video_path)
    global_cap = cv2.VideoCapture(input_path)
    video_width = int(global_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(global_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture_thread = Thread(target=video_capture, args=(frame_queue, image_queue))
    inference_thread = Thread(target=inference, args=(
    image_queue, detections_queue, fps_queue, dims, network, class_names, class_colors, detection_thresh))
    drawing_thread = Thread(target=drawing, args=(
    frame_queue, detections_queue, fps_queue, video_height, video_width, class_colors, output_filename))

    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()

    capture_thread.join()
    inference_thread.join()
    drawing_thread.join()
    print("Done with video")


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture_thread = Thread(target=video_capture, args=(frame_queue, darknet_image_queue))
    inference_thread = Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue))
    drawing_thread = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue))

    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()

    capture_thread.join()
    inference_thread.join()
    drawing_thread.join()
    print("Done with video")
