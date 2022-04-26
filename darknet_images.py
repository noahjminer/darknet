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

# CHANGES BEGIN HERE ------------------------------------------------------

# slices image according to dimensions, returns array of images
def sliceImagecv2(dimension, image):
    slices = []
    for dim in dimension:
        new_slice = image[dim[2]:dim[3], dim[0]:dim[1]]
        slices.append(new_slice)
    return slices


# baseline for image.
def image_detection_list(image_path, dims, network, class_names, class_colors, thresh):
    width = []
    height = []
    for dim in dims:
        width.append(dim[1] - dim[0])
        height.append(dim[3] - dim[2])

    orig_img = cv2.imread(image_path)
    bboxes = []
    images = sliceImagecv2(dims, orig_img)

    for i, img in enumerate(images):
        darknet_image = darknet.make_image(width[i], height[i], 3)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        bboxes.append(detections)
        darknet.free_image(darknet_image)

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


# Baseline on frame from video
def image_detection_list_on_frame(frame, dims, network, class_names, class_colors, thresh):
    width = []
    height = []

    for dim in dims:
        width.append(dim[1] - dim[0])
        height.append(dim[3] - dim[2])

    bboxes = []
    images = sliceImagecv2(dims, frame)
    # process slices
    for i, img in enumerate(images):
        darknet_image = darknet.make_image(width[i], height[i], 3)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        bboxes.append(detections)
        darknet.free_image(darknet_image)

    # correct bounding box coordinates
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


# instead of opening file, takes frame. Detects on frame.
def depth_detection_on_frame(frame, dims, network, class_names, class_colors, slice_side_length, detection_thresh):
    shape = frame.shape
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    images = []
    # resize slices
    for dim in dims:
        new_slice = frame[dim[2]:dim[3], dim[0]:dim[1]]
        if slice_side_length != 608:
            new_slice = cv2.resize(new_slice, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)

    bboxes = []
    darknet_image = darknet.make_image(width, height, 3)
    for image in images:
        darknet.copy_image_from_bytes(darknet_image, image.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=detection_thresh)
        bboxes.append(detections)
    darknet.free_image(darknet_image)

    # correcting coordinates of slice detections
    for i, img_boxes in enumerate(bboxes):
        for j, box in enumerate(img_boxes):
            bbox = box[2]  # detections are a tuple of (label, confidence, bbox)
            x, y, w, h = bbox
            x = dims[i][0] + (x / 608 * (dims[i][1] - dims[i][0]))
            y = dims[i][2] + (y / 608 * (dims[i][3] - dims[i][2]))
            bboxes[i][j] = (box[0], box[1], (x, y, w, h))

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

    flat_detections = [item for sub_list in bboxes for item in sub_list]

    final_detections = []
    for i, item in enumerate(flat_detections):  # removing non person labels for clarity
        if item[0] != 'person':
            continue
        final_detections.append(item)

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)
    return final_detections


# Depth based detection on a single image.
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

    images = []
    for dim in dims:
        new_slice = image_rgb[dim[2]:dim[3], dim[0]:dim[1]]
        slice_shape = new_slice.shape

        if slice_side_length != 608:
            new_slice = cv2.resize(new_slice, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)

    # correct bounding box coordinates
    bboxes = []
    if len(dims):
        decompress_slice_x = slice_shape[1] / 608
        decompress_slice_y = slice_shape[0] / 608
        bboxes = batch_detection(network, images, class_names, class_colors, batch_size=len(dims))

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
    decompress_rate_x = shape[1] / width
    decompress_rate_y = shape[0] / height
    darknet_image = darknet.make_image(width, height, 3)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=img_thresh)

    # correct bounding box coordinates
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
    # filter for people, remove you don't want it
    for i, item in enumerate(flat_detections):
        if item[0] != 'person':
            continue
        final_detections.append(item)

    final_detections = darknet.non_max_suppression_fast(final_detections, 0.7)
    print('---------------')
    elapsed_time = time.time() - prev_time
    print('Detection took ', elapsed_time)
    print('---------------')
    image = darknet.draw_slices(dims, orig_img, class_colors)
    return darknet.draw_boxes(final_detections, image, class_colors), final_detections, dims

# END OF CHANGES


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height = 608
    image_width = 608
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

