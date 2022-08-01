"""
This is the python script version of Semi-Global Matching with numpy.
Very little explanations are provided for the methods in this script.
To see a detailed overview of this numpy implementation see the jupyter
notebook: "sgm_numpy_notebook.ipynb".
"""

import argparse
import sys
import time as t
import cv2
import numpy as np


def get_path_cost(slice, offset, penalties, other_dim, disparity_dim):
    """
    Calculates the minimum costs for all potential disparities of 
    the pixels along a single path direction.
    Arguments:
        - slice: Array containing costs for all disparities, D, 
            along a direction, M, with dimension M x D
        - offset: Number of pixels on the border to ignore.
        - penalties: Matrix containing the penalties to assign to the 
            previous disparities costs. For previous disparities that differ 
            from current disparities. 
        - other_dim: Number of pixels in the current paths direction.
        - disparity_dim: Number of disparities to calculate minimum costs.

    Returns: The pixels minimum costs for all disparities, D, 
        along path direction, M, with shape M x D.
    """
    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for pixel_index in range(offset, other_dim):
        # Get all the minimum disparities costs from the previous pixel in the path
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        # Get all the disparities costs (from the cost volume) for the current pixel
        current_cost = slice[pixel_index, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        # Add penalties to the previous pixels disparities that differ from current pixels disparities
        costs = costs + penalties
        # Find minimum costs for the current pixels disparities using the previous disparities costs + penalties 
        costs = np.amin(costs, axis=0)  
        # Current pixels disparities costs + minimum previous pixel disparities costs (with penalty) - 
        # (constant term) minimum previous cost from all disparities 
        pixel_direction_costs = current_cost + costs - np.amin(previous_cost)
        minimum_cost_path[pixel_index, :] = pixel_direction_costs

    return minimum_cost_path   


def get_penalties(max_disparity, P2, P1):
    """
    Creates a matrix of all the potential penalties for matching
    a current disparity (represented by the column index), with 
    a previous disparity (represented by the row index).
    Arguments:
        - max_disparity: Maximum disparity of the array.
        - P2: Penalty for disparity difference > 1
        - P1: Penalty for disparity difference = 1
    
    Return: Matrix containing all the penalties when disparity d1 from a column
            is matched with a previous disparity d2 from the row.
    """
    p2 = np.full(shape=(max_disparity, max_disparity), fill_value=P2, dtype=np.uint32)
    p1 = np.full(shape=(max_disparity, max_disparity), fill_value=P1 - P2, dtype=np.uint32)
    p1 = np.tril(p1, k=1) # keep values lower than k'th diagonal
    p1 = np.triu(p1, k=-1) # keep values higher than k'th diagonal
    no_penalty = np.identity(max_disparity, dtype=np.uint32) * -P1 # create diagonal matrix with values -p1
    penalties = p1 + p2 + no_penalty
    return penalties


def aggregate_costs(cost_volume, P2, P1, height, width, disparities):
    """
    Calculates the pixels costs for all disparities along all paths (4 in this case).

    Arguments: 
        - cost_volume: Array containing the matching cost for each pixel at each disparity.
        - P2: Penalty for disparity difference > 1
        - P1: Penalty for disparity difference = 1
        - height: Number of rows of the image.
        - width: Number of columns of the image.
        - disparities: Number of disparities to calculate minimum matching costs.

    Returns: Array containing the pixels matching costs for all disparities along 
        all directions, with dimension H x W x D X 4.
    """
    sys.stdout.flush()
    dawn = t.time()

    penalties = get_penalties(disparities, P2, P1)

    print("\tProcessing North and South aggregation...")
    south_aggregation = np.zeros(shape=(height, width, disparities), dtype=np.uint32)
    north_aggregation = np.copy(south_aggregation)

    for x in range(0, width):
        # Takes all the rows and disparities for a single column
        south = cost_volume[:, x, :]
        # Invert the rows to get the opposite direction
        north = np.flip(south, axis=0)
        south_aggregation[:, x, :] = get_path_cost(south, 1, penalties, height, disparities)
        north_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, penalties, height, disparities), axis=0)


    print("\tProcessing East and West aggregation...", end='')
    east_aggregation = np.copy(south_aggregation)
    west_aggregation = np.copy(south_aggregation)
    for y in range(0, height):
        # Takes all the column and disparities for a single row
        east = cost_volume[y, :, :]
        # Invert the columns to get the opposite direction
        west = np.flip(east, axis=0)
        east_aggregation[y, :, :] = get_path_cost(east, 1, penalties, width, disparities)
        west_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, penalties, width, disparities), axis=0)

    # Combine the costs from all paths into a single aggregation volume
    aggregation_volume = np.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


def compute_census_np(left, right, csize, height, width):
    """
    Calculate census bit strings for each pixel in the left and right images.
    Arguments:
        - left: left grayscale image.
        - right: right grayscale image.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right images with pixel intensities replaced with census bit strings.
    """
    cheight = csize[0]
    cwidth = csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    left_census_values = np.pad(np.array([[
        np.where((left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)] - 
        np.full(shape=(cheight, cwidth), fill_value=left[y, x], dtype=np.int32)) < 0, 1, 0).flatten().dot(1 << np.arange(cheight * cwidth)[::-1])
        for x in range(x_offset, width - x_offset)]
        for y in range(y_offset, height - y_offset)]), 
            pad_width=((y_offset, y_offset), (x_offset, x_offset)), constant_values=0)


    right_census_values = np.pad(np.array([[
        np.where((right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)] - 
        np.full(shape=(cheight, cwidth), fill_value=right[y, x], dtype=np.int32)) < 0, 1, 0).flatten().dot(1 << np.arange(cheight * cwidth)[::-1])
        for x in range(x_offset, width - x_offset)] 
        for y in range(y_offset, height - y_offset)]), 
            pad_width=((y_offset, y_offset), (x_offset, x_offset)), constant_values=0)


    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_census_values, right_census_values


def compute_census(left, right, csize, height, width):
    """
    Calculate census bit strings for each pixel in the left and right images.
    Arguments:
        - left: left grayscale image.
        - right: right grayscale image.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right images with pixel intensities replaced with census bit strings.
    """
    cheight = csize[0]
    cwidth = csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # offset is used since pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            # left
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int32)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            # If value is less than center value assign 1 otherwise assign 0 
            left_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            left_census_pixel = np.int32(left_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1])) 
            left_census_values[y, x] = left_census_pixel

            # right
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int32)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            # If value is less than center value assign 1 otherwise assign 0 
            right_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            right_census_pixel = np.int32(right_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1])) 
            right_census_values[y, x] = right_census_pixel

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_census_values, right_census_values
    

def compute_costs_np_slow(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    Create cost volume for all potential disparities. 
    Cost volumes for both left and right images are calculated.
    Hamming distance is used to calculate the matching cost between 
    two pixels census values.
    Arguments:
        - left_census_values: left image containing census bit strings for each pixel (in integer form).
        - right_census_values: right image containing census bit strings for each pixel (in integer form).
        - max_disparity: maximum disparity to measure.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right cost volumes with dimensions H x W x D.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)
    disparity = max_disparity

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()

    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int32)

    def calc_hamming_distance(binary_number):
        return np.sum(np.frombuffer(np.binary_repr(binary_number, width=64).encode(), dtype='S1').astype(int))
    calc_hamming_distance_vec = np.vectorize(calc_hamming_distance)


    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int32(np.bitwise_xor(np.int32(left_census_values), rcensus))
        left_distance = calc_hamming_distance_vec(left_xor)
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int32(np.bitwise_xor(np.int32(right_census_values), lcensus))
        right_distance = calc_hamming_distance_vec(right_xor)
        right_cost_volume[:, :, d] = right_distance


    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def compute_costs(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    Create cost volume for all potential disparities. 
    Cost volumes for both left and right images are calculated.
    Hamming distance is used to calculate the matching cost between 
    two pixels census values.
    Arguments:
        - left_census_values: left image containing census bit strings for each pixel (in integer form).
        - right_census_values: right image containing census bit strings for each pixel (in integer form).
        - max_disparity: maximum disparity to measure.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right cost volumes with dimensions H x W x D.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int32)

    for d in range(0, max_disparity):
        # The right image is shifted d pixels accross
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        # 1 is assigned when the bits differ and 0 when they are the same
        left_xor = np.int32(np.bitwise_xor(np.int32(left_census_values), rcensus))
        # All the 1's are summed up to give us the number of different pixels (the cost)
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        # All the costs for that disparity are added to the cost volume
        left_cost_volume[:, :, d] = left_distance

        # The left image is shifted d pixels accross
        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        # 1 is assigned when the bits differ and 0 when they are the same
        right_xor = np.int32(np.bitwise_xor(np.int32(right_census_values), lcensus))
        # All the 1's are summed up to give us the number of different pixels (the cost)
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        # All the costs for that disparity are added to the cost volume
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume):
    """
    Converts the aggregation volume into a disparity map using 
    the winner takes all strategy. 
    Cost volume is first calculated by taking the sum of the costs over all paths.
    Then the disparities are determined by finding the 
    disparity index with the lowest cost for the pixel.

    Arguments:
        - aggregation_volume: Array containing the matching costs for 
            all pixels at all disparities and paths, with dimension H x W x D x N

    Returns: Disparity map with dimensions H x W.
    """
    # sum up costs for all directions
    volume = np.sum(aggregation_volume, axis=3) 
    # returns the disparity index with the minimum cost associated with each h x w pixel
    disparity_map = np.argmin(volume, axis=2) 
    return disparity_map


def normalize(disp, max_disparity):
    """
    Normalizes the disparity map, then
    quantizes it so that it can be displayed. 
    Arguments:
        - disp: disparity map with dimensions H x W.
        - max_disparity: maximum disparity of the array.
    
    Return: normalized then quantized array, ready for visualization.
    """
    return 255.0 * disp / max_disparity


def get_recall(disparity, gt, max_disparity):
    """
    Calculates the percentage of pixels from the 
    disparity map "disparity" within 3 absolute 
    disparity of the ground truth "gt". 
    Higher percentage is better.

    Arguments:
        - disparity: 
        - gt: 
        - max_disparity: 

    Returns: Percentage of pixels within 3 disparity of 
    the groundtruth.
    """
    gt = np.float32(gt)
    gt = np.int16(gt / 255.0 * float(max_disparity))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(max_disparity))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size



def sgm(left, right, max_disparity, P1, P2, csize, bsize):

    print("Performing Gaussian blur on the images...")
    left = cv2.GaussianBlur(left, bsize, 0, 0)
    right = cv2.GaussianBlur(right, bsize, 0, 0)

    print('\nStarting cost computation...')
    left_census, right_census = compute_census(left, right, csize, height, width)
    left_cost_volume, right_cost_volume = compute_costs(left_census, right_census, max_disparity, csize, height, width)


    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, P2, P1, height, width, max_disparity)
    print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, P2, P1, height, width, max_disparity)


    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), max_disparity))
    right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), max_disparity))


    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, bsize[0])

    return left_disparity_map, right_disparity_map



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='cones/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='cones/disp6.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    parser.add_argument('--p1', default=10, type=int, help='penalty for disparity difference = 1')
    parser.add_argument('--p2', default=120, type=int, help='penalty for disparity difference > 1')
    parser.add_argument('--csize', default=[5, 5], nargs="+", type=int, help='size of the kernel for the census transform')
    parser.add_argument('--bsize', default=[3, 3], nargs="+", type=int, help='size of the kernel for blurring the images and median filtering')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    save_images = args.images
    evaluation = args.eval
    max_disparity = args.disp
    P1 = args.p1
    P2 = args.p2
    csize = args.csize
    bsize = args.bsize

    dawn = t.time()

    print('\nLoading images...')
    left = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)
    height = left.shape[0]
    width = left.shape[1]
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert max_disparity > 0, 'maximum disparity must be greater than 0.'


    left_disparity_map, right_disparity_map = sgm(left, right, max_disparity, P1, P2, csize, bsize)


    if save_images:
        cv2.imwrite(f'left_{output_name}', np.array(left_disparity_map))
        cv2.imwrite(f'right_{output_name}', np.array(right_disparity_map))

    if evaluation:
        left_gt = cv2.imread(left_gt_name, cv2.IMREAD_GRAYSCALE)
        right_gt = cv2.imread(right_gt_name, cv2.IMREAD_GRAYSCALE)

        print('\nEvaluating left disparity map...')
        recall = get_recall(left_disparity_map, left_gt, max_disparity)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
        print('\nEvaluating right disparity map...')
        recall = get_recall(right_disparity_map, right_gt, max_disparity)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))

    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))