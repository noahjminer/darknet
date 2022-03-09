import cv2
import math


# Finds avg distance of a slice
# current equation is sum of red and green value / 510, but it isn't working as well as it seems
# Maybe we can find min and max and normalize?? idk
def find_avg_distance(image, dim):
    total = 0
    count = 0
    for i in range(dim[2], dim[3]):
        for j in range(dim[0], dim[1]):
            dist = (image[i][j][0] + image[i][j][1]) / 510
            total += dist
            count += 1
    return total / count if count != 0 else total / 1


# compresses depth image and finds avg for all slices in a 6x8 grid. Creates the dimensions as well
# hard coded to 6x8 with reasoning that resolution will be ~3000x2000
def create_depth_map(threshold, image_path, compression_rate):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    shape = image.shape
    height = int(shape[0] * compression_rate)
    width = int(shape[1] * compression_rate)

    dsize = (width, height)
    image_resized = cv2.resize(image, dsize)
    height = image_resized.shape[0]
    width = image_resized.shape[1]
    slice_dim_x = int(width / 8)
    slice_dim_y = int(height / 6)

    avgs = []
    for x in range(1, 9):
        col_avg = []
        for y in range(1, 7):
            dim = [(x-1)*slice_dim_x, x*slice_dim_x, (y-1)*slice_dim_y, y*slice_dim_y]
            avg = find_avg_distance(image, dim)
            col_avg.append(avg)
        avgs.append(col_avg)

    for i in avgs:
        print(i)


create_depth_map(.2, './saturation2compressed_disp.jpeg', .4)










