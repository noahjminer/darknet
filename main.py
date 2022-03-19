from darknet_images import depth_detection_list, image_detection_list
import argparse


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
    parser.add_argument('--slicesize', type=int,
                        help='slice size in pixels (square)',
                        default=500)
    parser.add_argument('--threshold', type=float,
                        help='depth threshold',
                        default=.15)
    return parser.parse_args()


def baseline_image(args):
    print('Beginning baseline')
    slice_size = args.slicesize


def baseline_video(args):
    print('Baseline: Video')


def depth(args):
    print('Depth')
    # Calibration
    # Detection


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'baseline':
        baseline(args)
    elif args.method == 'depth':
        depth(args)



