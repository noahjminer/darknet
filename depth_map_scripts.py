import cv2
import math
import numpy as np
import time



# counts valid pixels in a slice, and compares to proportion threshold
def create_depth_mask(image, dim, depth_thresh, proportion_thresh):
    count = 0
    for i in range(dim[2], dim[3]):
        for j in range(dim[0], dim[1]):
            dist = int(int((image[i][j])[0] * 100 / 255))
            if dist < depth_thresh:
                count += 1
    count = count / ((dim[3] - dim[2]) * (dim[1] - dim[0]))
    print(count, dim)
    if count > proportion_thresh:
        return True
    return False


# returns dimensions that satisfy threshold values
def create_depth_map_with_threshold(image_path, depth_thresh, proportion_thresh=.3, slice_side_length=608, expand_ratio=.05):
    prev_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    shape = image.shape
    height = int(shape[0])
    width = int(shape[1])

    # 608 is what darknet compresses images to
    num_slice_x = math.floor(width / slice_side_length)
    num_slice_y = math.floor(height / slice_side_length)

    slice_dim_x = math.ceil(width / num_slice_x)
    slice_dim_y = math.ceil(height / num_slice_y)

    avgs = []
    no_comp_dims = []
    expand_amount = expand_ratio * slice_side_length
    half_expand_amount = math.floor(expand_amount * .5)
    for x in range(1, num_slice_x + 1):
        col_avg = []
        for y in range(1, num_slice_y + 1):
            left = (x - 1) * slice_dim_x
            right = x * slice_dim_x
            top = (y - 1) * slice_dim_y
            bottom = y * slice_dim_y
            if bottom >= height:
                bottom = height - 1
                top = bottom - slice_dim_y
            if right >= width:
                right = width - 1
                left = right - slice_dim_x
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            dim = [left, right, top, bottom]
            if create_depth_mask(image, dim, depth_thresh, proportion_thresh):
                # for index, num in enumerate(dim):
                # dim[index] = int(num * decompress_rate)
                no_comp_dims.append(dim)

    expand_amount = int(2 * half_expand_amount)
    if len(no_comp_dims) > 1:
        for index, dim in enumerate(no_comp_dims):
            if dim[0] == 0:
                dim[1] += expand_amount
            elif dim[1] >= width - 1:
                dim[0] -= expand_amount
            else:
                dim[0] -= half_expand_amount
                dim[1] += half_expand_amount

            if dim[2] == 0:
                dim[3] += expand_amount
            elif dim[3] >= height - 1:
                dim[2] -= expand_amount
            else:
                dim[2] -= half_expand_amount
                dim[3] += half_expand_amount
            if dim[0] < 0:
                dim[0] = 0
            if dim[1] >= width:
                dim[1] = width-1
            if dim[2] < 0:
                dim[2] = 0
            if dim[3] >= height:
                dim[3] = height-1
    print(no_comp_dims)

    print('---------------')
    elapsed_time = time.time() - prev_time
    print('Depth map took ', elapsed_time)
    print('---------------')
    return no_comp_dims











