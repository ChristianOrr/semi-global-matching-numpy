"""
python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

author: David-Alexandre Beaupre
date: 2019/07/12
"""

import argparse
import sys
import time as t

from functools import partial
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def load_images(left_name, right_name, bsize):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, bsize, 0, 0)
    return left, right


@partial(jax.jit, static_argnames=["offset", "other_dim", "disparity_dim"])
def get_path_cost_jax(slice, offset, penalties, other_dim, disparity_dim):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    minimum_cost_path = jnp.zeros(shape=(other_dim, disparity_dim), dtype=jnp.uint32)
    minimum_cost_path.at[offset - 1, :].set(slice[offset - 1, :])

    for pixel_index in range(offset, other_dim):
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        current_cost = slice[pixel_index, :]
        costs = jnp.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = costs + penalties # add penalties for previous disparities differing from current disparities
        costs = jnp.amin(costs, axis=0) # find minimum costs for the disparities from the previous disparities costs plus penalties 
        pixel_direction_costs = current_cost + costs - jnp.amin(previous_cost)
        minimum_cost_path.at[pixel_index, :].set(pixel_direction_costs)

    return minimum_cost_path

@partial(jax.jit, static_argnames=["offset", "other_dim", "disparity_dim"])
def get_path_cost_jax_concat(slice, offset, penalties, other_dim, disparity_dim):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    minimum_cost_path = slice[offset - 1, :][None, :]

    for pixel_index in range(offset, other_dim):
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        current_cost = slice[pixel_index, :]
        costs = jnp.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = costs + penalties # add penalties for previous disparities differing from current disparities
        costs = jnp.amin(costs, axis=0) # find minimum costs for the disparities from the previous disparities costs plus penalties 
        pixel_direction_costs = current_cost + costs - jnp.amin(previous_cost)
        minimum_cost_path = jnp.concatenate((minimum_cost_path, pixel_direction_costs[None, :]), axis=0)

    return minimum_cost_path


def get_path_cost(slice, offset, penalties, other_dim, disparity_dim):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for pixel_index in range(offset, other_dim):
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        current_cost = slice[pixel_index, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = costs + penalties # add penalties for previous disparities differing from current disparities
        costs = np.amin(costs, axis=0) # find minimum costs for the disparities from the previous disparities costs plus penalties 
        pixel_direction_costs = current_cost + costs - np.amin(previous_cost)
        minimum_cost_path[pixel_index, :] = pixel_direction_costs

    return minimum_cost_path


@partial(jax.jit, static_argnames=["P2", "P1", "disparity_dim"], backend="cpu")
def get_penalties_jax(disparity_dim, P2, P1):

    p2 = jnp.full(shape=(disparity_dim, disparity_dim), fill_value=P2, dtype=jnp.uint32)
    p1 = jnp.full(shape=(disparity_dim, disparity_dim), fill_value=P1 - P2, dtype=jnp.uint32)
    p1 = jnp.tril(p1, k=1) # keep values lower than k'th diagonal
    p1 = jnp.triu(p1, k=-1) # keep values higher than k'th diagonal
    no_penalty = jnp.identity(disparity_dim, dtype=jnp.uint32) * -P1 # create diagonal matrix with values -p1
    penalties = p1 + p2 + no_penalty
    return penalties

def get_penalties(disparity_dim, P2, P1):

    p2 = np.full(shape=(disparity_dim, disparity_dim), fill_value=P2, dtype=np.uint32)
    p1 = np.full(shape=(disparity_dim, disparity_dim), fill_value=P1 - P2, dtype=np.uint32)
    p1 = np.tril(p1, k=1) # keep values lower than k'th diagonal
    p1 = np.triu(p1, k=-1) # keep values higher than k'th diagonal
    no_penalty = np.identity(disparity_dim, dtype=np.uint32) * -P1 # create diagonal matrix with values -p1
    penalties = p1 + p2 + no_penalty
    return penalties



def aggregate_costs(cost_volume, P2, P1, height, width, disparities):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (4 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    penalties = get_penalties(disparities, P2, P1)

    print("Processing North and South aggregation")
    south_aggregation = np.zeros(shape=(height, width, disparities), dtype=jnp.uint32)
    north_aggregation = np.copy(south_aggregation)

    for x in range(0, width):
        south = cost_volume[:, x, :]
        north = np.flip(south, axis=0)
        south_aggregation[:, x, :] = get_path_cost(south, 1, penalties, height, disparities)
        north_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, penalties, height, disparities), axis=0)


    print("Processing East and West aggregation.")
    east_aggregation = np.copy(south_aggregation)
    west_aggregation = np.copy(south_aggregation)
    for y in range(0, height):
        east = cost_volume[y, :, :]
        west = np.flip(east, axis=0)
        east_aggregation[y, :, :] = get_path_cost(east, 1, penalties, width, disparities)
        west_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, penalties, width, disparities), axis=0)

    aggregation_volume = np.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)
    
    return aggregation_volume

@partial(jax.jit, static_argnames=["P2", "P1", "height", "width", "disparities"], backend="cpu")
def aggregate_costs_jax(cost_volume, P2, P1, height, width, disparities):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (4 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    penalties = get_penalties_jax(disparities, P2, P1)

    print("Processing North and South aggregation")
    south_aggregation = jnp.zeros(shape=(height, width, disparities), dtype=jnp.uint32)
    north_aggregation = jnp.copy(south_aggregation)

    for x in range(0, width):
        south = cost_volume[:, x, :]
        north = jnp.flip(south, axis=0)
        south_aggregation.at[:, x, :].set(get_path_cost_jax(south, 1, penalties, height, disparities))
        north_aggregation.at[:, x, :].set(jnp.flip(get_path_cost_jax(north, 1, penalties, height, disparities), axis=0))


    print("Processing East and West aggregation.")
    east_aggregation = jnp.copy(south_aggregation)
    west_aggregation = jnp.copy(south_aggregation)
    for y in range(0, height):
        east = cost_volume[y, :, :]
        west = jnp.flip(east, axis=0)
        east_aggregation.at[y, :, :].set(get_path_cost_jax(east, 1, penalties, width, disparities))
        west_aggregation.at[y, :, :].set(jnp.flip(get_path_cost_jax(west, 1, penalties, width, disparities), axis=0))

    aggregation_volume = jnp.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)

    return aggregation_volume

@partial(jax.jit, static_argnames=["P2", "P1", "height", "width", "disparities"], backend="cpu")
def aggregate_costs_jax_concat(cost_volume, P2, P1, height, width, disparities):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (4 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    penalties = get_penalties_jax(disparities, P2, P1)

    print("Processing North and South aggregation")
    south_aggregation = get_path_cost_jax_concat(cost_volume[:, 0, :], 1, penalties, height, disparities)[:, None, :]
    north_aggregation = get_path_cost_jax_concat(jnp.flip(cost_volume[:, 0, :], axis=0), 1, penalties, height, disparities)[:, None, :]
    for x in range(1, width):
        south = cost_volume[:, x, :]
        north = jnp.flip(south, axis=0)
        south_aggregation = jnp.concatenate((south_aggregation, get_path_cost_jax_concat(south, 1, penalties, height, disparities)[:, None, :]), axis=1)
        north_aggregation = jnp.concatenate((north_aggregation, jnp.flip(get_path_cost_jax_concat(north, 1, penalties, height, disparities), axis=0)[:, None, :]), axis=1)

    print("Processing East and West aggregation.")
    east_aggregation = get_path_cost_jax_concat(cost_volume[0, :, :], 1, penalties, width, disparities)[None, ...]
    west_aggregation = get_path_cost_jax_concat(jnp.flip(cost_volume[0, :, :], axis=0), 1, penalties, width, disparities)[None, ...]
    for y in range(1, height):
        east = cost_volume[y, :, :]
        west = jnp.flip(east, axis=0)
        east_aggregation = jnp.concatenate((east_aggregation, get_path_cost_jax_concat(east, 1, penalties, width, disparities)[None, ...]), axis=0)
        west_aggregation = jnp.concatenate((west_aggregation, jnp.flip(get_path_cost_jax_concat(west, 1, penalties, width, disparities), axis=0)[None, ...]), axis=0)

    aggregation_volume = jnp.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)

    return aggregation_volume

@partial(jax.jit, static_argnames=["csize", "height", "width"], backend="gpu")
def compute_census_jax(left, right, csize, height, width):
    cheight = csize[0]
    cwidth = csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    left_census_values = jnp.pad(jnp.array([[
        jnp.where((left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)] - 
        jnp.full(shape=(cheight, cwidth), fill_value=left[y, x], dtype=jnp.int32)) < 0, 1, 0).flatten().dot(1 << jnp.arange(cheight * cwidth)[::-1])
        for x in range(x_offset, width - x_offset)]
        for y in range(y_offset, height - y_offset)]), 
            pad_width=((y_offset, y_offset), (x_offset, x_offset)), constant_values=0)


    right_census_values = jnp.pad(jnp.array([[
        jnp.where((right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)] - 
        jnp.full(shape=(cheight, cwidth), fill_value=right[y, x], dtype=jnp.int32)) < 0, 1, 0).flatten().dot(1 << jnp.arange(cheight * cwidth)[::-1])
        for x in range(x_offset, width - x_offset)] 
        for y in range(y_offset, height - y_offset)]), 
            pad_width=((y_offset, y_offset), (x_offset, x_offset)), constant_values=0)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_census_values, right_census_values


def compute_census_np(left, right, csize, height, width):
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
    cheight = csize[0]
    cwidth = csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            # left
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference

            left_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            left_census_pixel = np.int64(left_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1]))
            left_census_values[y, x] = left_census_pixel

            # right
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference

            right_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            right_census_pixel = np.int64(right_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1])) 
            right_census_values[y, x] = right_census_pixel

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_census_values, right_census_values

@partial(jax.jit, static_argnames=["csize", "max_disparity", "width"], backend="gpu")
def compute_costs_jax(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()

    def calc_hamming_distance(binary_array):
        return jnp.sum(binary_array)
    calc_hamming_distance_vec = jnp.vectorize(calc_hamming_distance)

    left_cost_volume = jnp.array([ 
        calc_hamming_distance_vec(
                jnp.bitwise_xor(jnp.int32(left_census_values), 
                jnp.pad(right_census_values[:, x_offset:(width - d - x_offset)], pad_width=((0, 0), (x_offset + d, x_offset)), constant_values=0))
                )
        for d in range(max_disparity)], dtype=jnp.uint32)


    right_cost_volume = jnp.array([ 
        calc_hamming_distance_vec(
                jnp.bitwise_xor(jnp.int32(right_census_values), 
                jnp.pad(left_census_values[:, (x_offset + d):(width - x_offset)], pad_width=((0, 0), (x_offset, x_offset + d)), constant_values=0))
                )
        for d in range(max_disparity)], dtype=jnp.uint32)


    left_cost_volume = np.array(jnp.moveaxis(left_cost_volume, 0, -1))
    right_cost_volume = np.array(jnp.moveaxis(right_cost_volume, 0, -1))


    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume

def compute_costs_np(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)
    disparity = max_disparity

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()

    def calc_hamming_distance(binary_number):
        return np.sum(np.frombuffer(np.binary_repr(binary_number, width=64).encode(), dtype='S1').astype(int))
    calc_hamming_distance_vec = np.vectorize(calc_hamming_distance)

    left_cost_volume = np.array([ 
        calc_hamming_distance_vec(
            np.int32(
                np.bitwise_xor(np.int32(left_census_values), 
                np.pad(right_census_values[:, x_offset:(width - d - x_offset)], pad_width=((0, 0), (x_offset + d, x_offset)), constant_values=0)
                )
            ))
        for d in range(disparity)], dtype=np.uint32)


    right_cost_volume = np.array([ 
        calc_hamming_distance_vec(
            np.int32(
                np.bitwise_xor(np.int32(right_census_values), 
                np.pad(left_census_values[:, (x_offset + d):(width - x_offset)], pad_width=((0, 0), (x_offset, x_offset + d)), constant_values=0)
                )
            ))
        for d in range(disparity)], dtype=np.uint32)


    left_cost_volume = np.moveaxis(left_cost_volume, 0, -1) 
    right_cost_volume = np.moveaxis(right_cost_volume, 0, -1)


    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume
    

def compute_costs(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)
    disparity = max_disparity

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)

    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: H x W disparity image.
    """
    volume = jnp.sum(aggregation_volume, axis=3) # sum up costs for all directions
    disparity_map = jnp.argmin(volume, axis=2) # returns the disparity index with the minimum cost associated with each h x w pixel
    return disparity_map


def normalize(volume, max_disparity):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / max_disparity


def get_recall(disparity, gt, args):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param args: program arguments.
    :return: rate of correct predictions.
    """
    gt = jnp.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    gt = jnp.int16(gt / 255.0 * float(args.disp))
    disparity = jnp.int16(jnp.float32(disparity) / 255.0 * float(args.disp))
    correct = jnp.count_nonzero(jnp.abs(disparity - gt) <= 3)
    return float(correct) / gt.size


def median_blur(image, filter_size):
    num_pixels = filter_size * filter_size
    kernel = jnp.full((filter_size, filter_size), fill_value=1/num_pixels)
    smoothed_image = jsp.signal.convolve(image, kernel, mode="same")
    return smoothed_image


def sgm():
    """
    main function applying the semi-global matching algorithm.
    :return: void.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='cones/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='cones/disp6.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    save_images = args.images
    evaluation = args.eval
    max_disparity = args.disp
    P1 = 10 # penalty for disparity difference = 1
    P2 = 120 # penalty for disparity difference > 1
    csize = (7, 7) # size of the kernel for the census transform.
    bsize = (3, 3) # size of the kernel for blurring the images and median filtering.

    dawn = t.time()

    print('\nLoading images...')
    left, right = load_images(left_name, right_name, bsize)
    height = left.shape[0]
    width = left.shape[1]
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert max_disparity > 0, 'maximum disparity must be greater than 0.'

    print('\nStarting cost computation...')
    left_census, right_census = compute_census_jax(left, right, csize, height, width)

    left_cost_volume, right_cost_volume = compute_costs_jax(left_census, right_census, max_disparity, csize, height, width)
    if save_images:
        cv2.imwrite('left_census.png', np.uint8(left_census))
        left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), max_disparity))
        cv2.imwrite('disp_map_left_cost_volume.png', left_disparity_map)
        cv2.imwrite('right_census.png', np.uint8(right_census))
        right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), max_disparity))
        cv2.imwrite('disp_map_right_cost_volume.png', right_disparity_map)

    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, P2, P1, height, width, max_disparity)
    print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, P2, P1, height, width, max_disparity)

    print('\nSelecting best disparities...')
    left_disparity_map = jnp.uint8(normalize(select_disparity(left_aggregation_volume), max_disparity))
    right_disparity_map = jnp.uint8(normalize(select_disparity(right_aggregation_volume), max_disparity))
    if save_images:
        cv2.imwrite('left_disp_map_no_post_processing.png', left_disparity_map)
        cv2.imwrite('right_disp_map_no_post_processing.png', right_disparity_map)

    print('\nApplying median filter...')
    left_disparity_map = median_blur(left_disparity_map, bsize[0])
    right_disparity_map = median_blur(right_disparity_map, bsize[0])
    cv2.imwrite(f'left_{output_name}', np.array(left_disparity_map))
    cv2.imwrite(f'right_{output_name}', np.array(right_disparity_map))

    if evaluation:
        print('\nEvaluating left disparity map...')
        recall = get_recall(left_disparity_map, left_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
        print('\nEvaluating right disparity map...')
        recall = get_recall(right_disparity_map, right_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))

    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))


if __name__ == '__main__':
    sgm()
