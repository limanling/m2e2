'''
Revised based on https://github.com/hassanhub/MultiGrounding/blob/master/code/utils.py
'''

from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# bbox generation config
rel_peak_thr = .3
rel_rel_thr = .3
ioa_thr = .6
topk_boxes = 3


def heat2bbox(heat_map, original_image_shape):
    h, w = heat_map.shape

    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(heat_map, exclude_border=False,
                                 threshold_rel=rel_peak_thr)  # find local peaks of heat map

    heat_resized = cv2.resize(heat_map, (
    original_image_shape[1], original_image_shape[0]))  ## resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) *
                           np.asarray([original_image_shape]) /
                           np.asarray([[h, w]])
                           ).astype('int32')

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask)
        l = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value)  # you can change to pk_value * probability of sentence matching image or etc.

    ## Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[:min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)):
        if i not in to_remove:
            bounding_boxes.append({
                'score': box_scores[i],
                'bbox': bboxes[i],
                'bbox_normalized': np.asarray([
                    bboxes[i][0] / heat_resized.shape[1],
                    bboxes[i][1] / heat_resized.shape[0],
                    bboxes[i][2] / heat_resized.shape[1],
                    bboxes[i][3] / heat_resized.shape[0],
                ]),
            })

    return bounding_boxes


def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False,
                       bboxes=[], order=None, show=True):
    thr_hit = 1  # a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60  # the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)

    if len(bboxes) > 0:  # it gets normalized bbox
        if order == None:
            order = 'xxyy'

        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order == 'xxyy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[1] * W), int(bbox_norm[2] * H), int(
                    bbox_norm[3] * H)
            elif order == 'xyxy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[2] * W), int(bbox_norm[1] * H), int(
                    bbox_norm[3] * H)
            x_length, y_length = x_max - x_min, y_max - y_min
            box = plt.Rectangle((x_min, y_min), x_length, y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name != '':
                ax.text(x_min + .5 * x_length, y_min + 10, en_name,
                        verticalalignment='center', horizontalalignment='center',
                        # transform=ax.transAxes,
                        color='white', fontsize=15)
                # an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                # plt.gca().add_patch(an)

    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)

    # plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    # plt.figure(3, figsize=(6, 6))
    plt.subplot(1, 3, 3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def filter_bbox(bbox_dict, order=None):
    thr_fit = .99  # the biggest acceptable bbox should not exceed 80% of the image
    if order == None:
        order = 'xxyy'

    filtered_bbox = []
    filtered_bbox_norm = []
    filtered_score = []
    if len(bbox_dict) > 0:  # it gets normalized bbox
        for i in range(len(bbox_dict)):
            bbox = bbox_dict[i]['bbox']
            bbox_norm = bbox_dict[i]['bbox_normalized']
            bbox_score = bbox_dict[i]['score']
            if order == 'xxyy':
                x_min, x_max, y_min, y_max = bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3]
            elif order == 'xyxy':
                x_min, x_max, y_min, y_max = bbox_norm[0], bbox_norm[2], bbox_norm[1], bbox_norm[3]
            if bbox_score > 0:
                x_length, y_length = x_max - x_min, y_max - y_min
                if x_length * y_length < thr_fit:
                    filtered_score.append(bbox_score)
                    filtered_bbox.append(bbox)
                    filtered_bbox_norm.append(bbox_norm)
    return filtered_bbox, filtered_bbox_norm, filtered_score


def crop_resize_im(image, bbox, size, order='xxyy'):
    H, W, _ = image.shape
    if order == 'xxyy':
        roi = image[int(bbox[2] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[1] * W), :]
    elif order == 'xyxy':
        roi = image[int(bbox[1] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[2] * W), :]
    roi = cv2.resize(roi, size)
    return roi


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def isCorrect(bbox_annot, bbox_pred, iou_thr=.4):
    iou_value_max = 0.0
    for bbox_p in bbox_pred:
        for bbox_a in bbox_annot:
            iou_value = IoU(bbox_p, bbox_a)
            iou_value_max = max(iou_value, iou_value_max)
            if iou_value >= iou_thr:
                return 1, iou_value
    return 0, iou_value_max


def isCorrectHit(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)
    print('max_loc', max_loc)
    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1
    return 0


def check_percent(bboxes):
    for bbox in bboxes:
        x_length = bbox[2] - bbox[0]
        y_length = bbox[3] - bbox[1]
        if x_length * y_length < .05:
            return False
    return True


def union(bbox):
    if len(bbox) == 0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox, axis=0)
    mins = np.min(bbox, axis=0)
    return [[mins[0], mins[1], maxes[2], maxes[3]]]


def attCorrectness(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    h_s = np.sum(heatmap_resized)
    if h_s == 0:
        return 0
    else:
        heatmap_resized /= h_s
    att_correctness = 0
    for bbox in bbox_annot:
        x0, y0, x1, y1 = bbox
        att_correctness += np.sum(heatmap_resized[y0:y1, x0:x1])
    return att_correctness


def calc_correctness(annot, heatmap, orig_img_shape, iou_thr=.5):
    bbox_dict = heat2bbox(heatmap, orig_img_shape)
    bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = annot['bbox']
    bbox_norm_pred = union(bbox_norm)
    # print('bbox_norm_annot', bbox_norm_annot)
    # print('bbox_norm_pred', bbox_norm_pred)
    # print('bbox_annot', bbox_annot)
    # print('bbox_norm', bbox_norm)
    bbox_correctness, bbox_iou = isCorrect(bbox_norm_annot, bbox_norm_pred, iou_thr=iou_thr)
    hit_correctness = isCorrectHit(bbox_annot, heatmap, orig_img_shape)
    att_correctness = attCorrectness(bbox_annot, heatmap, orig_img_shape)
    return bbox_correctness, hit_correctness, att_correctness, bbox_iou

def precision_bbox(boxA, boxB):
    '''

    :param boxA: predicted
    :param boxB: GT
    :return:
    '''
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    # boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    precision = interArea / float(boxAArea)
    # recall = interArea / float(boxB)

    # return the intersection over union value
    return precision

def calc_correctness_box(annot_all_norm, bbox_one_norm, iou_thr=.5, count_inside=False):
    '''

    :param bbox_annot: list of boxes
    :param bbox_pred_one: one bbox
    :return:
    '''
    # bbox_norm_annot = annot['bbox_norm']
    # bbox_annot = annot['bbox']

    # bbox_norm = bboxes_role['bbox_norm']
    # bbox = bboxes_role['bbox']

    # for bbox_p in bbox_norm:
    iou_value_max = 0.0
    for bbox_a in annot_all_norm:
        iou_value = IoU(bbox_one_norm, bbox_a)
        iou_value_max = max(iou_value, iou_value_max)
        if iou_value >= iou_thr:
            return 1, iou_value
        if count_inside:
            precision = precision_bbox(bbox_one_norm, bbox_a)
            if precision >= 0.9:
                return 1, precision
    return 0, iou_value_max

def calc_correctness_box_uion(annot_all_norm, bbox_all_norm, iou_thr=.5, count_inside=False):
    '''

    :param bbox_annot: list of boxes
    :param bbox_pred_one: one bbox
    :return:
    '''
    # bbox_norm_annot = annot['bbox_norm']
    # bbox_annot = annot['bbox']

    # bbox_norm = bboxes_role['bbox_norm']
    # bbox = bboxes_role['bbox']

    # for bbox_p in bbox_norm:
    # iou_value_max = 0.0
    annot_all_norm_union = union(annot_all_norm)
    bbox_all_norm_union = union(bbox_all_norm)

    for bbox_a in annot_all_norm_union:
        for bbox_b in bbox_all_norm_union:
            iou_value = IoU(bbox_a, bbox_b)
            if iou_value >= iou_thr:
                return 1, iou_value
            if count_inside:
                precision = precision_bbox(bbox_b, bbox_a)
                if precision >= 0.9:
                    return 1, precision
    return 0, iou_value
