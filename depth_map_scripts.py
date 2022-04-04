import cv2
import math
import numpy as np
import time

# Finds avg distance of a slice
# current equation is sum of red and green value / 510, but it isn't working as well as it seems
# Maybe we can find min and max and normalize?? idk
c1 = np.array([0, 0, 0])
c2 = np.array([255, 255, 255])


# need to check bounds
def find_avg_distance(image, dim):
    total = 0
    count = 0
    for i in range(dim[2], dim[3]):
        for j in range(dim[0], dim[1]):
            dist = int(int((image[i][j] - c1)[0] * 100 / 255))
            total += dist
            count += 1
    return total / count if count != 0 else total / 1


def create_depth_mask(image, dim, depth_thresh, proportion_thresh):
    count = 0
    for i in range(dim[2], dim[3]):
        for j in range(dim[0], dim[1]):
            dist = int(int((image[i][j] - c1)[0] * 100 / 255))
            if dist < depth_thresh:
                count += 1
    count = count * 1.0 / ((dim[3] - dim[2]) * (dim[1] - dim[0]))
    print(count, dim)
    if count > proportion_thresh:
        return True
    return False


def create_depth_map_with_threshold(image_path, depth_thresh, proportion_thresh=.3, expand_ratio=.05):
    prev_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    shape = image.shape
    height = int(shape[0])
    width = int(shape[1])

    # 608 is what darknet compresses images to
    num_slice_x = math.floor(width / 608)
    num_slice_y = math.floor(height / 608)

    slice_dim_x = math.ceil(width / num_slice_x)
    slice_dim_y = math.ceil(height / num_slice_y)

    avgs = []
    no_comp_dims = []
    top_overlap = 0
    bottom_overlap = 0
    left_overlap = 0
    right_overlap = 0
    expand_amount = expand_ratio * 608
    half_expand_amount = math.floor(expand_amount * .5)
    for x in range(1, num_slice_x + 1):
        col_avg = []
        for y in range(1, num_slice_y + 1):
            left = (x - 1) * slice_dim_x
            right = x * slice_dim_x
            top = (y - 1) * slice_dim_y
            bottom = y * slice_dim_y
            if bottom > height:
                bottom = height - 1
                top = bottom - slice_dim_y
            if right > width:
                right = width - 1
                left = right - slice_dim_x
            dim = [left, right, top, bottom]
            if create_depth_mask(image, dim, depth_thresh, proportion_thresh):
                # for index, num in enumerate(dim):
                # dim[index] = int(num * decompress_rate)
                no_comp_dims.append(dim)

    print(no_comp_dims)
    expand_amount = int(2 * half_expand_amount)
    for index, dim in enumerate(no_comp_dims):
        if dim[0] == 0:
            dim[1] += expand_amount
        elif dim[1] == width - 1:
            dim[0] -= expand_amount
        else:
            dim[0] -= half_expand_amount
            dim[1] += half_expand_amount

        if dim[2] == 0:
            dim[3] += expand_amount
        elif dim[3] == height - 1:
            dim[2] -= expand_amount
        else:
            dim[2] -= half_expand_amount
            dim[3] += half_expand_amount
    print(no_comp_dims)

    print('---------------')
    elapsed_time = time.time() - prev_time
    print('Depth map took ', elapsed_time)
    print('---------------')
    return no_comp_dims


# compresses depth image and finds avg for all slices in a 6x8 grid. Creates the dimensions as well
# hard coded to 6x8 with reasoning that resolution will be ~3000x2000
# prob shouldnt compress here and also could have smarter dimensions
def create_depth_map_with_avg(no_comp_thresh, image_path, compression_rate):
    prev_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    decompress_rate = 1 / compression_rate

    shape = image.shape
    height = int(shape[0] * compression_rate)
    width = int(shape[1] * compression_rate)

    dsize = (width, height)
    image_resized = cv2.resize(image, dsize)
    height = image_resized.shape[0]
    width = image_resized.shape[1]
    # Smarter dimensions needed

    slice_dim_x = int(width / 8)
    slice_dim_y = int(height / 6)

    avgs = []
    no_comp_dims = []
    for x in range(1, 9):
        col_avg = []
        for y in range(1, 7):
            # need to do bound checking here
            left = (x - 1) * slice_dim_x
            right = x * slice_dim_x
            top = (y - 1) * slice_dim_y
            bottom = y * slice_dim_y
            if bottom > height:
                bottom = height - 1
            if right > width:
                right = width - 1
            dim = [left, right, top, bottom]
            avg = find_avg_distance(image, dim)
            if avg < no_comp_thresh:
                for index, num in enumerate(dim):
                    dim[index] = int(num * decompress_rate)
                no_comp_dims.append(dim)
            col_avg.append(avg)
        avgs.append(col_avg)
    print('---------------')
    elapsed_time = time.time() - prev_time
    print('Depth map took ', elapsed_time)
    print('---------------')
    return no_comp_dims
    # for i in avgs:
    #     print(i)











