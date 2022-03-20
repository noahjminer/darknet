import cv2
import math
import numpy as np
import time


# Finds avg distance of a slice
# current equation is sum of red and green value / 510, but it isn't working as well as it seems
# Maybe we can find min and max and normalize?? idk
c1 = np.array([0, 0, 0])
c2 = np.array([255, 255, 0])


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


# compresses depth image and finds avg for all slices in a 6x8 grid. Creates the dimensions as well
# hard coded to 6x8 with reasoning that resolution will be ~3000x2000
# prob shouldnt compress here and also could have smarter dimensions
def create_depth_map(no_comp_thresh, image_path, compression_rate):
    prev_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    decompress_rate = 1/compression_rate

    shape = image.shape
    height = int(shape[0] * compression_rate)
    width = int(shape[1] * compression_rate)

    dsize = (width, height)
    image_resized = cv2.resize(image, dsize)
    height = image_resized.shape[0]
    width = image_resized.shape[1]
    # Smarter dimensions needed
    slice_dim_x = int(width / 5)
    slice_dim_y = int(height / 8)

    avgs = []
    no_comp_dims = []
    for x in range(1, 9):
        col_avg = []
        for y in range(1, 7):
            # need to do bound checking here
            dim = [(x-1)*slice_dim_x, x*slice_dim_x, (y-1)*slice_dim_y, y*slice_dim_y]
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











