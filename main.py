from darknet_images import depth_detection_list, image_detection_list
from monodepth2.test_simple import create_depth_image
import darknet
import argparse
import random
import os
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(
        'Run different solutions / models with argument parse.'
    )
    parser.add_argument('--image_path', type=str, help='path to image/video, if batch is true, then directory of images')
    parser.add_argument('--batch', type=bool,
                        help='If true, run on group of images / videos in sequence.',
                        default=False)
    parser.add_argument('--method', type=str,
                        help='name of method',
                        choices=['baseline', 'depth'],
                        default='baseline')
    parser.add_argument('--video', type=bool, help='video or image? default to image',
                       default=False)
    parser.add_argument('--slice_size', type=int,
                        help='slice size in pixels (square)',
                        default=500)
    parser.add_argument('--depth_threshold', type=float,
                        help='depth threshold',
                        default=15)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],
                        default="mono+stereo_640x192")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    return parser.parse_args()


def baseline(args):
    print('Baseline')


def generate_depth_image(args, file):
    prev = time.time()
    print_update(f'Generating depth image for {file}')
    output = create_depth_image(args, file)
    diff = time.time() - prev
    print_update(f'Generation complete in {diff}, output at {output}')
    return output


def depth(args):
    files = []

    assert args.image_path is not None

    if args.batch:
        files = get_path_list_from_dir(args.image_path)
    else:
        files.append(args.image_path)

    # create darknet things
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    print(class_colors)

    if args.video:
        for f in files:
            generate_depth_image(args, f)
    else:
        for f in files:
            depth_path = generate_depth_image(args, f)
            image, detections = depth_detection_list(f, network, class_names, class_colors, depth_path, 15, .25)
            cv2.imwrite('result.jpg', image)


# ----------- utils ------------


def get_path_list_from_dir(path):
    files = os.listdir(path)
    for index, file in enumerate(files):
        files[index] = os.path.join(path, file)
    return files


def print_update(msg):
    print('----------------------')
    print(msg)
    print('----------------------')


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'baseline':
        baseline(args)
    elif args.method == 'depth':
        depth(args)



