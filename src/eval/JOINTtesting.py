import numpy as np
from skimage.feature import peak_local_max
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt, mpld3
import matplotlib.patches as patches
import os
from collections import defaultdict
# import ujson as json
import json
import sys
sys.setrecursionlimit(10000000)

from src.util.util_img import calc_correctness, rel_peak_thr, rel_rel_thr, ioa_thr, topk_boxes
from src.util.util_img import calc_correctness_box, calc_correctness_box_uion
from src.dataflow.numpy.anno_mapping import event_type_norm, role_name_norm
from src.util import consts


class JointTester():
    def __init__(self, ignore_place_sr, ignore_time):
        # bbox generation config
        # self.ioa_thr = 0.9
        # self.topk_boxes = 300
        # self.rel_peak_thr = 0.9
        # self.rel_rel_thr = 0.3
        self.iou_thr = 0.5
        self.ignore_place_sr = ignore_place_sr
        # self.ignore_time = ignore_time



    def calculate_report(self, data, image_path, visual_path, add_object,
                         keep_events_sr=0):
        '''

        :param vision_result:
        :param method: att or obj
        :return:
        '''

        # evt_num_gt = 0
        # evt_num_pred = 0
        # evt_correct = 0
        #
        # role_att_iou_correct = 0
        # role_att_hit_correct = 0
        # role_att_cor_correct = 0
        # role_obj_iou_correct = 0
        # role_num_gt = 0
        # role_num_pred = 0

        all_str = 'all'
        evt_num_gt = defaultdict(int)
        evt_num_pred = defaultdict(int)
        evt_correct = defaultdict(int)

        role_att_iou_correct = defaultdict(int)
        role_att_hit_correct = defaultdict(int)
        role_att_cor_correct = defaultdict(int)
        role_obj_iou_correct = defaultdict(int)
        role_obj_iou_union_correct = defaultdict(int)
        role_num_gt = defaultdict(int)
        role_num_pred = defaultdict(int)
        role_num_gt_union = defaultdict(int)
        role_num_pred_union = defaultdict(int)

        evt_p = defaultdict(float)
        evt_r = defaultdict(float)
        evt_f1 = defaultdict(float)
        
        if visual_path is not None:
            visual_html = os.path.join(visual_path, 'image_result.html')
            visual_html_writer = open(visual_html, 'w')
            if not os.path.exists(os.path.join(visual_path, 'image_heatmaps')):
                os.makedirs(os.path.join(visual_path, 'image_heatmaps'), exist_ok=True)
            visual_json_writer = open(os.path.join(visual_path, 'image_result.json'), 'w')

        print('start generate report', len(data))
        for imgid in data:
            if keep_events_sr > 0 and 'ground_truth' not in data[imgid]:
                continue
            if 'ground_truth' in data[imgid]:
                event_type_gt = event_type_norm(data[imgid]['ground_truth']['event_type'])
            else:
                event_type_gt = consts.O_LABEL_NAME
            if keep_events_sr > 0 and event_type_gt == consts.O_LABEL_NAME:
                continue

            img_file = os.path.join(image_path, imgid)
            if not os.path.exists(img_file):
                img_file = os.path.join(image_path, imgid+'.png')
            img = cv2.imread(img_file)
            img = img[:, :, ::-1]
            h, w, _ = img.shape

            if 'event_type' in data[imgid]:
                event_type_pred = event_type_norm(data[imgid]['event_type'])
            else:
                event_type_pred = consts.O_LABEL_NAME
            verb_pred = data[imgid]['verb']
            if event_type_pred != consts.O_LABEL_NAME:
                evt_num_pred[event_type_pred] += 1
                evt_num_pred[all_str] += 1
            if event_type_gt != consts.O_LABEL_NAME:
                evt_num_gt[event_type_gt] += 1
                evt_num_gt[all_str] += 1
            if event_type_pred != consts.O_LABEL_NAME and event_type_pred == event_type_gt:
                evt_correct[event_type_pred] += 1
                evt_correct[all_str] += 1

            if visual_path is not None:
                # visual_json = defaultdict(lambda : defaultdict())
                # visual_json[imgid]['sr_verb'] = verb_pred
                # visual_json[imgid]['event_type_pred'] = event_type_pred
                # # visual_json[imgid]['event_type_gt'] = event_type_gt
                # visual_json[imgid]['role_pred'] = defaultdict(lambda : defaultdict())
                visual_json = dict()
                visual_json['image_id']=imgid
                visual_json['sr_verb'] = verb_pred
                visual_json['event_type_pred'] = event_type_pred
                # visual_json]['event_type_gt'] = event_type_gt
                visual_json['role_pred'] = dict()

                visual_html_writer.write(
                    "event_pred: %s (%s); event_gt: %s; <br>" % (event_type_pred, verb_pred, event_type_gt))
                ### visualization has how many subplots
                num_figs = len(data[imgid]['role']) if 'role' in data[imgid] else 1
                # if len(data[imgid]['role']) == 0:
                #     num_figs = 1
                    # for item in data[imgid]['ground_truth']['role'].values():
                #     num_figs += len(item)
                if 'ground_truth' in data[imgid] and 'role' in data[imgid]['ground_truth']:
                    num_figs += len(data[imgid]['ground_truth']['role'])
                    # for role_gt in data[imgid]['ground_truth']['role']:
                    #     # for item in data[imgid]['ground_truth']['role'].values():
                    #     # num_figs += len(item) # visualization all bboxes in the same one for each role
                    #     num_figs += len(data[imgid]['ground_truth']['role'][role_gt])
                fig, axes = plt.subplots(1, num_figs, figsize=(5 * num_figs, 5), squeeze=False)
                # print('num_figs', num_figs, len(axes))
                i = 0
                ### visualization has how many subplots


            # get ground truth role
            if 'ground_truth' in data[imgid] and 'role' in data[imgid]['ground_truth']:
                for role_gt in data[imgid]['ground_truth']['role'].keys():
                    if self.ignore_place_sr and role_gt.upper() == 'PLACE':
                        continue
                    # print('role_gt', role_gt)

                    if not add_object:
                        role_num_gt[role_gt] += 1
                        role_num_gt[all_str] += 1
                    else:
                        role_num_gt[role_gt] += len(data[imgid]['ground_truth']['role'][role_gt])
                        role_num_gt[all_str] += len(data[imgid]['ground_truth']['role'][role_gt])
                        role_num_gt_union[role_gt] += 1
                        role_num_gt_union[all_str] += 1
                    annos_bbox = dict()
                    annos_bbox['bbox'] = list()
                    annos_bbox['bbox_norm'] = list()
                    # heatmaps_pred = list()

                    if visual_path is not None:
                        ### visualization all bboxes in the same one for each role
                        # print('i', i, 'num_figs', num_figs, len(axes))
                        axes[0][i].imshow(img)
                        axes[0][i].set_title(f'{role_gt}') # axes[0][i].set_title(f'{role_gt}: {noun_gt}')
                        axes[0][i].get_xaxis().set_visible(False)
                        axes[0][i].get_yaxis().set_visible(False)

                    for noun_gt, x1, y1, x2, y2 in data[imgid]['ground_truth']['role'][role_gt]:
                        box_gt = [x1, y1, x2, y2]
                        box_gt_norm = [float(x1)/float(w), float(y1)/float(h), float(x2)/float(w), float(y2)/float(h)]
                        annos_bbox['bbox'].append(box_gt)
                        annos_bbox['bbox_norm'].append(box_gt_norm)

                        if visual_path is not None:
                            axes[0][i].add_patch(patches.Rectangle(
                                (box_gt[0], box_gt[1]), (box_gt[2] - box_gt[0]), (box_gt[3] - box_gt[1]),
                                linewidth=1, edgecolor='r', facecolor='none'
                            ))

                    if visual_path is not None:
                        i += 1

                    if not add_object:
                        # get the recall
                        if role_gt in data[imgid]['heatmap']:
                            noun_pred = data[imgid]['heatmap'][role_gt]
                            heatmap_pred = data[imgid]['heatmap'][role_gt]
                            # heatmaps_pred.append(heatmap_pred)

                            bbox_correctness, hit_correctness, att_correctness, bbox_iou = self.calculate_att(annos_bbox, heatmap_pred, (h, w))
                            if bbox_correctness == 1:
                                role_att_iou_correct[role_gt] += 1
                                role_att_iou_correct[all_str] += 1
                            if hit_correctness == 1:
                                role_att_hit_correct[role_gt] += 1
                                role_att_hit_correct[all_str] += 1
                            if att_correctness > 0.5:
                                role_att_cor_correct[role_gt] += 1
                                role_att_cor_correct[all_str] += 1
                    else:
                        if role_gt in data[imgid]['role']:
                            obj_list = data[imgid]['role'][role_gt]
                            bbox_norm_list = list()
                            for obj_current_id, obj_current_bbox, obj_current_label in obj_list:
                                obj_current_bbox_norm =  [float(obj_current_bbox[0])/float(w),
                                                          float(obj_current_bbox[1])/float(h),
                                                          float(obj_current_bbox[2])/float(w),
                                                          float(obj_current_bbox[3])/float(h)]
                                bbox_norm_list.append(obj_current_bbox_norm)
                                bbox_correctness, bbox_iou = self.calculate_obj(annos_bbox, obj_current_bbox_norm)
                                # print(imgid, obj_current_id, role_gt, bbox_correctness, bbox_iou)
                                if bbox_correctness == 1:
                                    role_obj_iou_correct[role_gt] += 1
                                    role_obj_iou_correct[all_str] += 1
                            bbox_correctness_union, bbox_iou = self.calculate_obj_union(annos_bbox, bbox_norm_list)
                            if bbox_correctness_union == 1:
                                role_obj_iou_union_correct[role_gt] += 1
                                role_obj_iou_union_correct[all_str] += 1

            # get the predicted role
            if 'role' in data[imgid]:
                if not add_object:
                    for role, noun in data[imgid]['role'].items():
                        if self.ignore_place_sr and role.upper() == 'PLACE':
                            continue
                        role_num_pred[role] += 1
                        role_num_pred[all_str] += 1

                        if visual_path is not None:
                            # visualization:
                            heatmap = data[imgid]['heatmap'][role]
                            # self.visualize_heatmap(heatmap, img, w, h, event_type_pred, role, noun, axes[0][i])
                            peak_coords = peak_local_max(heatmap, exclude_border=False, threshold_rel=rel_peak_thr)
                            heatmap = cv2.resize(heatmap, (w, h))
                            peak_coords_resized = ((peak_coords + 0.5) *
                                                np.asarray([[h, w]]) /
                                                np.asarray([[7, 7]])
                                                ).astype('int32')

                            bboxes = []
                            box_scores = []
                            for pk_coord in peak_coords_resized:
                                pk_value = heatmap[tuple(pk_coord)]
                                mask = heatmap > pk_value * rel_rel_thr
                                labeled, n = ndi.label(mask)
                                l = labeled[tuple(pk_coord)]
                                yy, xx = np.where(labeled == l)
                                min_x = np.min(xx)
                                min_y = np.min(yy)
                                max_x = np.max(xx)
                                max_y = np.max(yy)
                                bboxes.append((min_x, min_y, max_x, max_y))
                                box_scores.append(pk_value)

                            box_idx = np.argsort(-np.asarray(box_scores))
                            box_idx = box_idx[:min(topk_boxes, len(box_scores))]
                            bboxes = [bboxes[j] for j in box_idx]

                            # visual_json[imgid]['role_pred'][role]['noun'] = noun
                            # visual_json[imgid]['role_pred'][role]['bbox'] = bboxes
                            visual_json['role_pred'][role]={ 'noun' : noun, 'bbox': bboxes}

                            # print('i', i)
                            axes[0][i].imshow(img)
                            axes[0][i].imshow(heatmap, alpha=.7)

                            axes[0][i].set_title('Pred:%s_%s: %s' % (event_type_pred, role, noun))

                            for box in bboxes:
                                axes[0][i].add_patch(patches.Rectangle(
                                    (box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]),
                                    linewidth=1, edgecolor='r', facecolor='none'
                                ))
                            axes[0][i].get_xaxis().set_visible(False)
                            axes[0][i].get_yaxis().set_visible(False)
                            i += 1
                else:
                    for role_pred in data[imgid]['role']:
                        
                        if len(role_pred) == 0:
                            continue
                        if self.ignore_place_sr and role_pred.upper()  == 'PLACE':
                            continue
                        role_num_pred[role_pred] += len(data[imgid]['role'][role_pred])
                        role_num_pred[all_str] += len(data[imgid]['role'][role_pred])


                        if visual_path is not None:
                            visual_json['role_pred'][role_pred] = list()
                            # visualization:
                            axes[0][i].imshow(img)
                            axes[0][i].set_title(f'Pred:{role_pred}')  # axes[0][i].set_title(f'{role_gt}: {noun_gt}')
                            axes[0][i].get_xaxis().set_visible(False)
                            axes[0][i].get_yaxis().set_visible(False)
                            for obj_current_id, obj_current_bbox, obj_current_label in data[imgid]['role'][role_pred]:
                                axes[0][i].add_patch(patches.Rectangle(
                                    (obj_current_bbox[0], obj_current_bbox[1]),
                                    (obj_current_bbox[2] - obj_current_bbox[0]),
                                    (obj_current_bbox[3] - obj_current_bbox[1]),
                                    linewidth=1, edgecolor='r', facecolor='none'
                                ))

                                # visual_json[imgid]['role_pred'][role_pred]['noun'] = obj_current_label
                                # visual_json[imgid]['role_pred'][role_pred]['bbox'] = obj_current_bbox
                                # visual_json[imgid]['role_pred'][role_pred] = {'noun':obj_current_label,'bbox':obj_current_bbox}
                                visual_json['role_pred'][role_pred].append(obj_current_bbox)
                            i += 1

            # save the pic of no roles:
            if ('role' not in data[imgid] and 'ground_truth' not in data[imgid]) or (len(data[imgid]['role']) == 0):# and len(data[imgid]['role']) == 0):
                if visual_path is not None:
                    # visualization:
                    axes[0][0].imshow(img)
                    axes[0][0].set_title(f'Pred:{verb_pred}') 
                    axes[0][0].get_xaxis().set_visible(False)
                    axes[0][0].get_yaxis().set_visible(False)

            if visual_path is not None:
                # for every image
                # mpld3.save_html(fig, visual_path.replace('.html', '/'+imgid+'.html'))
                # mpld3.show()
                image_result_save_path = os.path.join(visual_path, 'image_heatmaps', imgid.replace(' ', '_')+'.png')
                plt.savefig(image_result_save_path)
                visual_html_writer.write(
                    '<img src=\"./image_heatmaps/'+imgid.replace(' ', '_')+'.png'+'\" width=\"100%\">\n<br><br>\n')
                visual_json_writer.write('%s\n' % (json.dumps(visual_json)))

        for type in evt_correct:
            evt_p[type] = float(evt_correct[type]) / float(evt_num_pred[type])
            evt_r[type] = float(evt_correct[type]) / float(evt_num_gt[type])
            evt_f1[type] = self.get_f1(evt_p[type], evt_r[type])

        role_scores = defaultdict(lambda : defaultdict(float))
        # print('role_scores', role_scores)
        for role_type in role_num_gt:
            if not add_object:
                role_scores['role_att_iou_p'][role_type] = float(role_att_iou_correct[role_type]) / (float(role_num_pred[role_type]) + 1e-6)
                role_scores['role_att_iou_r'][role_type] = float(role_att_iou_correct[role_type]) / (float(role_num_gt[role_type]) + 1e-6)
                role_scores['role_att_iou_f1'][role_type] = self.get_f1(role_scores['role_att_iou_p'][role_type], role_scores['role_att_iou_r'][role_type])

                role_scores['role_att_hit_p'][role_type] = float(role_att_hit_correct[role_type]) / (float(role_num_pred[role_type]) + 1e-6)
                role_scores['role_att_hit_r'][role_type] = float(role_att_hit_correct[role_type]) / (float(role_num_gt[role_type]) + 1e-6)
                role_scores['role_att_hit_f1'][role_type] = self.get_f1(role_scores['role_att_hit_p'][role_type], role_scores['role_att_hit_r'][role_type])

                role_scores['role_att_cor_p'][role_type] = float(role_att_cor_correct[role_type]) / (float(role_num_pred[role_type]) + 1e-6)
                role_scores['role_att_cor_r'][role_type] = float(role_att_cor_correct[role_type]) / (float(role_num_gt[role_type]) + 1e-6)
                role_scores['role_att_cor_f1'][role_type] = self.get_f1(role_scores['role_att_cor_p'][role_type], role_scores['role_att_cor_r'][role_type])
            else:
                role_scores['role_obj_iou_p'][role_type] = float(role_obj_iou_correct[role_type]) / (float(role_num_pred[role_type]) + 1e-6)
                role_scores['role_obj_iou_r'][role_type] = float(role_obj_iou_correct[role_type]) / (float(role_num_gt[role_type]) + 1e-6)
                role_scores['role_obj_iou_f1'][role_type] = self.get_f1(role_scores['role_obj_iou_p'][role_type], role_scores['role_obj_iou_r'][role_type])


                role_scores['role_obj_iou_union_p'][role_type] = float(role_obj_iou_union_correct[role_type]) / (
                            float(role_num_pred_union[role_type]) + 1e-6)
                role_scores['role_obj_iou_union_r'][role_type] = float(role_obj_iou_union_correct[role_type]) / (
                            float(role_num_gt_union[role_type]) + 1e-6)
                role_scores['role_obj_iou_union_f1'][role_type] = self.get_f1(role_scores['role_obj_iou_union_p'][role_type],
                                                                        role_scores['role_obj_iou_union_r'][role_type])

        # if visual_path is not None:
        #     visual_json_writer.write(json.dumps(visual_json, open(visual_json_path, 'w')))
            
        # print('role_scores', role_scores)
        return evt_p, evt_r, evt_f1, role_scores #role_att_iou_p, role_att_iou_r, role_att_iou_f1, \
                # role_att_hit_p, role_att_hit_r, role_att_hit_f1, \
               # role_att_cor_p, role_att_cor_r, role_att_cor_f1

    def get_f1(self, p, r):
        return 2.0 * p * r / (p + r + 1e-6)

    def visualize_heatmap(self, heatmap, img, w, h, event, role, noun, axe):
        peak_coords = peak_local_max(heatmap, exclude_border=False, threshold_rel=rel_peak_thr)
        heatmap = cv2.resize(heatmap, (w, h))
        peak_coords_resized = ((peak_coords + 0.5) *
                               np.asarray([[h, w]]) /
                               np.asarray([[7, 7]])
                               ).astype('int32')

        bboxes = []
        box_scores = []
        for pk_coord in peak_coords_resized:
            pk_value = heatmap[tuple(pk_coord)]
            mask = heatmap > pk_value * rel_rel_thr
            labeled, n = ndi.label(mask)
            l = labeled[tuple(pk_coord)]
            yy, xx = np.where(labeled == l)
            min_x = np.min(xx)
            min_y = np.min(yy)
            max_x = np.max(xx)
            max_y = np.max(yy)
            bboxes.append((min_x, min_y, max_x, max_y))
            box_scores.append(pk_value)

        box_idx = np.argsort(-np.asarray(box_scores))
        box_idx = box_idx[:min(topk_boxes, len(box_scores))]
        bboxes = [bboxes[i] for i in box_idx]

        axe.imshow(img)
        axe.imshow(heatmap, alpha=.7)

        axe.set_title('%s_%s: %s' % (event, role, noun))

        for box in bboxes:
            axe.add_patch(patches.Rectangle(
                (box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]),
                linewidth=1, edgecolor='r', facecolor='none'
            ))
        axe.get_xaxis().set_visible(False)
        axe.get_yaxis().set_visible(False)

        return axe


    def calculate_att(self, anno, heatmap, orig_img_shape):
        '''
        :param anno: dict
                annot['bbox_norm']:
                annot['bbox']: [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
        :param heatmap:
        :param orig_img_shape: h, w
        :param iou_thr:
        :return:
        '''
        bbox_correctness, hit_correctness, att_correctness, bbox_iou = calc_correctness(anno, heatmap, orig_img_shape, iou_thr=self.iou_thr)
        return bbox_correctness, hit_correctness, att_correctness, bbox_iou


    def calculate_obj(self, anno, bbox_one_norm):
        '''

        :param anno: dict
                annot['bbox_norm']:
                annot['bbox']: [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
        :param box_one: list of bbox
        :return:
        '''
        # ground truth object box and role type
        bbox_correctness, bbox_iou = calc_correctness_box(anno['bbox_norm'], bbox_one_norm, iou_thr=self.iou_thr, count_inside=True)
        return bbox_correctness, bbox_iou


    def calculate_obj_union(self, anno, bbox_all_norm):
        '''

        :param anno: dict
                annot['bbox_norm']:
                annot['bbox']: [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
        :param box_one: list of bbox
        :return:
        '''
        # ground truth object box and role type
        bbox_correctness, bbox_iou = calc_correctness_box_uion(anno['bbox_norm'], bbox_all_norm, iou_thr=self.iou_thr, count_inside=True)
        return bbox_correctness, bbox_iou
