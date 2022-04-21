import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet
import math

from depth_map_scripts import create_depth_map_with_threshold


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                             "txt with paths to them, or a folder. Image valid"
                             " formats are jpg, jpeg or png."
                             "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise (ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
               glob.glob(os.path.join(images_path, "*.png")) + \
               glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_resized = cv2.resize(image_rgb, (width, height),
        #                            interpolation=cv2.INTER_LINEAR)
        custom_image = image.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


# CHANGES BEGIN HERE ------------------------------------------------------

# Slices images to desired size in np.array dim (y, x, 3).
# This is the format that darknet accepts
# known problems:
#   If image size is just over slice width or height,
#   it cuts off images.


def sliceImagecv2(dimension, image):
    slices = []
    for dim in dimension:
        newSlice = image[dim[2]:dim[3], dim[0]:dim[1]]
        slices.append(newSlice)

    return slices


def image_detection_list(image_path, dims, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    # width = slice_width
    # height = slice_height
    width = []
    height = []
    for dim in dims:
        width.append(dim[1] - dim[0])
        height.append(dim[3] - dim[2])

    orig_img = cv2.imread(image_path)
    shape = orig_img.shape
    orig_img_width = shape[1]
    orig_img_height = shape[0]
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    bboxes = []
    images = sliceImagecv2(dims, orig_img)
    # orig_img_resized = cv2.resize(orig_img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    for i, img in enumerate(images):
        darknet_image = darknet.make_image(width[i], height[i], 3)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image_resized = cv2.resize(image_rgb, (width[i], height[i]), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        bboxes.append(detections)
        darknet.free_image(darknet_image)

    # CORRECT BOUNDING BOX COORDINATES
    # TODO: BOUNDING BOX COMBINATIONS
    # can edit once sizes are uniform, or collect array of sizes.

    for i, img_boxes in enumerate(bboxes):
        for j, box in enumerate(img_boxes):
            bbox = box[2]  # detections are a tuple of (label, confidence, bbox)
            x, y, w, h = bbox
            x = dims[i][0] + x
            y = dims[i][2] + y
            bboxes[i][j] = (box[0], box[1], (x, y, w, h))

    flat_detections = [item for sub_list in bboxes for item in sub_list]

    final_detections = []
    for i, item in enumerate(flat_detections):  # removing non person labels for clarity
        if item[0] != 'person':
            continue
        final_detections.append(item)

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)

    return darknet.draw_boxes(final_detections, orig_img, class_colors), final_detections


def image_detection_list_on_frame(frame, dims, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    # width = slice_width
    # height = slice_height
    width = []
    height = []
    for dim in dims:
        width.append(dim[1] - dim[0])
        height.append(dim[3] - dim[2])

    shape = frame.shape
    bboxes = []
    images = sliceImagecv2(dims, frame)
    # orig_img_resized = cv2.resize(orig_img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    for i, img in enumerate(images):
        darknet_image = darknet.make_image(width[i], height[i], 3)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image_resized = cv2.resize(image_rgb, (width[i], height[i]), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        bboxes.append(detections)
        darknet.free_image(darknet_image)

    # CORRECT BOUNDING BOX COORDINATES
    # TODO: BOUNDING BOX COMBINATIONS
    # can edit once sizes are uniform, or collect array of sizes.

    for i, img_boxes in enumerate(bboxes):
        for j, box in enumerate(img_boxes):
            bbox = box[2]  # detections are a tuple of (label, confidence, bbox)
            x, y, w, h = bbox
            x = dims[i][0] + x
            y = dims[i][2] + y
            bboxes[i][j] = (box[0], box[1], (x, y, w, h))

    flat_detections = [item for sub_list in bboxes for item in sub_list]

    final_detections = []
    for i, item in enumerate(flat_detections):  # removing non person labels for clarity
        if item[0] != 'person':
            continue
        final_detections.append(item)

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)

    return final_detections


def depth_detection_on_frame(frame, dims, network, class_names, class_colors, slice_side_length, detection_thresh):
    shape = frame.shape
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    images = []
    for dim in dims:
        new_slice = frame[dim[2]:dim[3], dim[0]:dim[1]]
        if slice_side_length != 608:
            new_slice = cv2.resize(new_slice, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)

    # batch_time = time.time()
    bboxes = []
    darknet_image = darknet.make_image(width, height, 3)
    for image in images:
        darknet.copy_image_from_bytes(darknet_image, image.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=detection_thresh)
        bboxes.append(detections)
    darknet.free_image(darknet_image)

    # can edit once sizes are uniform, or collect array of sizes.
    for i, img_boxes in enumerate(bboxes):
        for j, box in enumerate(img_boxes):
            bbox = box[2]  # detections are a tuple of (label, confidence, bbox)
            x, y, w, h = bbox
            x = dims[i][0] + (x / 608 * (dims[i][1] - dims[i][0]))
            y = dims[i][2] + (y / 608 * (dims[i][3] - dims[i][2]))
            bboxes[i][j] = (box[0], box[1], (x, y, w, h))

    # print('slices done in ', time.time() - batch_time)

    # do whole image
    # whole_time = time.time()
    decompress_rate_x = shape[1] / width
    decompress_rate_y = shape[0] / height
    darknet_image = darknet.make_image(width, height, 3)
    image_resized = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=detection_thresh)
    for index, detection in enumerate(detections):
        x, y, w, h = detection[2]
        x = x * decompress_rate_x
        y = y * decompress_rate_y
        w = w * decompress_rate_x
        h = h * decompress_rate_y
        detections[index] = (detection[0], detection[1], (x, y, w, h))

    bboxes.append(detections)
    darknet.free_image(darknet_image)
    # print(time.time() - whole_time, ' seconds for whole image')

    flat_detections = [item for sub_list in bboxes for item in sub_list]

    final_detections = []
    for i, item in enumerate(flat_detections):  # removing non person labels for clarity
        if item[0] != 'person':
            continue
        final_detections.append(item)

    # for outputting depth map variables
    # final_detections.append((thresh, compress_rate, (10, 10, 100, 10)))

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)
    # print('---------------')
    # # elapsed_time = time.time() - prev_time
    # print('Detection took ', elapsed_time)
    # print('---------------')
    # image = darknet.draw_slices(dims, orig_img, class_colors)
    return final_detections


def depth_detection_list(image_path, args, class_names, class_colors, depth_path, depth_thresh, img_thresh,
                         proportion_thresh, slice_side_length):
    dims = []
    if os.path.exists(image_path.split('.', 1)[0] + '_dims.txt') and args.no_refresh_dims:
        with open(image_path.split('.', 1)[0] + '_dims.txt', 'r') as infile:
            content = infile.readlines()
            for line in content:
                nums = [int(n) for n in line.split(' ')]
                dims.append(nums)
            slice_side_length = nums[1] - nums[0]
    else:
        dims = create_depth_map_with_threshold(depth_path, depth_thresh, proportion_thresh, slice_side_length)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=len(dims)
    )

    prev_time = time.time()
    orig_img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    shape = orig_img.shape

    batch_time = time.time()
    images = []
    for dim in dims:
        new_slice = image_rgb[dim[2]:dim[3], dim[0]:dim[1]]
        slice_shape = new_slice.shape

        if slice_side_length != 608:
            new_slice = cv2.resize(new_slice, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)

    bboxes = []
    if len(dims):
        decompress_slice_x = slice_shape[1] / 608
        decompress_slice_y = slice_shape[0] / 608
        bboxes = batch_detection(network, images, class_names, class_colors, batch_size=len(dims))

        # can edit once sizes are uniform, or collect array of sizes.
        for i, img_boxes in enumerate(bboxes):
            for j, box in enumerate(img_boxes):
                bbox = box[2]  # detections are a tuple of (label, confidence, bbox)
                x, y, w, h = bbox
                x = x * decompress_slice_x
                y = y * decompress_slice_y
                x = dims[i][0] + x
                y = dims[i][2] + y
                w = w * decompress_slice_x
                h = h * decompress_slice_y
                bboxes[i][j] = (box[0], box[1], (x, y, w, h))
    # do whole image
    whole_time = time.time()
    decompress_rate_x = shape[1] / width
    decompress_rate_y = shape[0] / height
    darknet_image = darknet.make_image(width, height, 3)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=img_thresh)
    for index, detection in enumerate(detections):
        x, y, w, h = detection[2]
        x = x * decompress_rate_x
        y = y * decompress_rate_y
        w = w * decompress_rate_x
        h = h * decompress_rate_y
        detections[index] = (detection[0], detection[1], (x, y, w, h))
    bboxes.append(detections)
    darknet.free_image(darknet_image)

    flat_detections = [item for sub_list in bboxes for item in sub_list]

    final_detections = []
    for i, item in enumerate(flat_detections):  # removing non person labels for clarity
        if item[0] != 'person':
            continue
        final_detections.append(item)

    # for outputting depth map variables
    # final_detections.append((thresh, compress_rate, (10, 10, 100, 10)))

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)
    print('---------------')
    elapsed_time = time.time() - prev_time
    print('Detection took ', elapsed_time)
    print('---------------')
    image = darknet.draw_slices(dims, orig_img, class_colors)
    return darknet.draw_boxes(final_detections, image, class_colors), final_detections, dims


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        predictions = darknet.decode_detection(predictions)
        # images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x / width, y / height, w / width, h / height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections, = batch_detection(network, images, class_names,
                                          class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)


def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    images = load_images(args.input)

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
        )
        elapsed = time.time() - prev_time

        save_path = image_name[:len(image_name) - 4] + "_result.jpg"
        cv2.imwrite(save_path, image)

        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        # darknet.print_detections(detections, args.ext_output)
        print(f'Processed in {elapsed} seconds.')
        # if not args.dont_show:
        #     cv2.imshow('Inference', image)
        #     if cv2.waitKey() & 0xFF == ord('q'):
        #         break
        # cv2.wait(0)
        index += 1


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
