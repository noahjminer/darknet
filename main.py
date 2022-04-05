from numpy.ma.core import fabs
from darknet_images import depth_detection_list, image_detection_list
from darknet_video import detect_video
from monodepth2.test_simple import create_depth_image_from_file, create_depth_image_from_frame
from depth_map_scripts import create_depth_map_with_threshold
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
    parser.add_argument('--image_path', type=str,
                        help='path to image/video, if batch is true, then directory of images')
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
                        help='depth threshold, with 0 is furthest ditance away',
                        default=15)
    parser.add_argument('--write_slice_dim_file', type=bool,
                        help='writes slices to txt file of same name',
                        default=False)
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
                        default="mono+stereo_1024x320")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--detection_thresh", type=float, default=.25,
                        help="remove detections with lower confidence, value from 0.0 to 1.0, with 0.0 is lowest certainty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--slice_side_length", default=608, type=int,
                        help="length of slice side in depth map")
    parser.add_argument("--refresh_dims", default=False, type=bool
                        )
    parser.add_argument("--refresh_depth", default=False, type=bool
                        )
    parser.add_argument("--proportion_thresh", type=float, default=.3,
                        help="the proportion of a slice that further than the depth_threshold, value from 0.0 to 1.0, with 0.0 is No area in the slice is further than threshold")
    return parser.parse_args()


def baseline(args):
    print('Baseline')


def generate_depth_image_from_file(args, f):
    prev = time.time()
    print_update(f'Generating depth image for {f}')
    output = create_depth_image_from_file(args, f)
    diff = time.time() - prev
    print_update(f'Generation complete in {diff}, output at {output}')
    return output


def generate_depth_image_from_frame(args, f):
    prev = time.time()
    print_update(f'Generating depth image for {f}')
    cap = cv2.VideoCapture(f)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    output = create_depth_image_from_frame(args, frame_rgb, f)
    diff = time.time() - prev
    print_update(f'Generation complete in {diff}, output at {output}')
    return output


def depth_mask(args):
    assert args.image_path is not None

    random.seed(3)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )


# Lower calibration time - do it in seperate process
def depth_slice(args):
    files = []

    assert args.image_path is not None
    if args.batch:
        files = get_path_list_from_dir(args.image_path)
    else:
        files.append(args.image_path)

    # create darknet things
    random.seed(3)  # deterministic bbox colors

    depth_thresh = args.depth_threshold
    detection_thresh = args.detection_thresh
    proportion_thresh = args.proportion_thresh

    depth_root_path = args.image_path.rsplit('.', 1)[0] + "_disp.jpeg"

    if args.video:
        for f in files:
            if not os.path.exists(depth_root_path) or args.refresh_depth:
                depth_root_path = generate_depth_image_from_frame(args, f)
            else:
                print_update('depth image already generated, moving on...')
            # generate dims
            dims = []
            if os.path.exists(f.split('.', 1)[0] + '_dims.txt') and not args.refresh_depth:
                with open(f.split('.', 1)[0] + '_dims.txt', 'r') as infile:
                    content = infile.readlines()
                    for line in content:
                        nums = [int(n) for n in line.split(' ')]
                        dims.append(nums)
                        args.slice_side_length = nums[1] - nums[0]
            else:
                dims = create_depth_map_with_threshold(depth_root_path, depth_thresh, proportion_thresh, args.slice_side_length)
            # pass into darknet_video func
            dims_path = args.image_path.rsplit('.', 1)[0] + "_dims.txt"
            if args.write_slice_dim_file:
                write_slice_file(dims, dims_path)
            detect_video(args, f, dims, detection_thresh)
    else:
        for index, f in enumerate(files):
            if not os.path.exists(depth_root_path) or args.refresh_depth:
                depth_root_path = generate_depth_image_from_file(args, f)
            else:
                print_update('depth image already generated, moving on...')
            image, detections, depth_dims = depth_detection_list(f, args, None, None, depth_root_path, depth_thresh, args.slice_side_length,
                                                                 detection_thresh, args.proportion_thresh)
            new_path = args.image_path.rsplit('.', 1)[0] + "_result." + args.image_path.rsplit('.', 1)[1]
            dims_path = args.image_path.rsplit('.', 1)[0] + "_dims.txt"
            cv2.imwrite(new_path, image)
            if args.write_slice_dim_file:
                write_slice_file(depth_dims, dims_path)


# ----------- utils ------------
def write_slice_file(dims, dims_path):
    with open(dims_path, 'w') as outfile:
        for dim in dims:
            outfile.write(str(dim[0]) + ' ' + str(dim[1]) + ' ' + str(dim[2]) + ' ' + str(dim[3]) + '\n')


def write_detections_to_file(detections, file_name):
    output = ""
    for detection in detections:
        _, _, bbox = detection
        output = output + f'{int(bbox[0])},{int(bbox[1])},{int(bbox[0]+bbox[2])},{int(bbox[1]+bbox[3])}\n'
    filename = file_name.split('/')[-1]
    filename = file_name.split('.')[0]
    filename = filename + '.txt'
    with open(filename, 'w') as outfile:
        outfile.write(output)


def get_path_list_from_dir(path):
    files = os.listdir(path)
    for index, f in enumerate(files):
        files[index] = os.path.join(path, f)
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
        depth_slice(args)
    elif args.method == 'depth_mask':
        depth_mask(args)



