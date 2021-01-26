import random
import argparse
import numpy as np
import json
import os
root = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]


def data_gather(data_path):
    data_path = os.path.join(root, 'data/custom', data_path)
    all_data = []
    direc_label = data_path+'/labels/'
    for i in os.listdir(direc_label):
        with open(direc_label+i, 'r') as fh:
            data = fh.readlines()
        for j in data:
            r_w, r_h = list(map(float, j.split()))[3:]
            all_data.append((r_w, r_h))
    return all_data


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, _ = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum/n


def print_anchors(centroids, resize):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    for i in sorted_indices:
        out_string += str(int(anchors[i, 0]*resize)) + \
            ',' + str(int(anchors[i, 1]*resize)) + ', '

    print(out_string[:-2])


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        # distances.shape = (ann_num, anchor_num)
        distances = np.array(distances)

        print("iteration {}: dists = {}".format(
            iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def main(annotation_dims, num_anchors, resize=416):
    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' %
          avg_IOU(annotation_dims, centroids))
    print_anchors(centroids, resize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default='fly', help="Dataset Name")
    parser.add_argument("--anchor", type=int, default=9,
                        help="Numbers of Anchors")
    parser.add_argument("--resize", type=int, default=416,
                        help="Image size after resizeing")

    opt = parser.parse_args()
    print(opt)
    all_data = data_gather(opt.dataset)
    main(all_data, opt.anchor, opt.resize)
