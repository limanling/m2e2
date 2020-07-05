import codecs
import os

import torch
import torch.utils.data as data
from PIL import Image
from collections import defaultdict
import json
import csv
import pickle
import traceback
import numpy as np

from src.util import consts

class ImSituDataset(data.Dataset):
    def __init__(self, image_dir, vocab_situation_noun, vocab_situation_role, vocab_situation_verb,
                 imsitu_ontology_file, imsitu_annotation_file, verb_mapping_file, role_mapping_file, object_ontology_file,
                 object_detection_pkl_file, object_detection_threshold=0.2, transform=None):
        self.image_dir = image_dir
        self.vocab_situation_noun = vocab_situation_noun
        self.vocab_situation_role = vocab_situation_role
        self.vocab_situation_verb = vocab_situation_verb

        # imsitu_info = json.load(open(os.path.join(imsitu_dir, "imsitu_space.json")))
        imsitu_info = json.load(open(imsitu_ontology_file))
        self.nouns = imsitu_info["nouns"]
        self.verbs = imsitu_info["verbs"]
        # self.annotation = json.load(open(os.path.join(imsitu_dir, "train.json")))
        self.annotation = json.load(open(imsitu_annotation_file))

        self.sr_mapping_verb = self.load_mapping_verb(verb_mapping_file)

        self.object_detection_threshold = object_detection_threshold
        self.object_label = self._get_labels(object_ontology_file)
        self.object_results = pickle.load(open(object_detection_pkl_file, 'rb'))

        # self.img_verbs, self.img_verb_roles, self.img_verb_role_num = self.__getobjects__(image_dir)
        self.img_verbs, self.img_verb_roles = self.__getobjects__(image_dir)
        self.idx_imgs = self.__getids__()
        self.ids = list(self.idx_imgs.keys())
        self.transform = transform

    def load_mapping_verb(self, verb_mapping_file):
        verb_type_dict = dict()
        for line in open(verb_mapping_file):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            verb_type_dict[tabs[0]] = tabs[1]
        return verb_type_dict

    def _get_labels(self, class_map_file):
        label_name = {}
        with open(class_map_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                label_name[row[0]] = row[1]
        return label_name

    # img -> patch & entity
    def __getobjects__(self, image_dir):
        img_verbs = defaultdict(lambda: str)
        img_verb_roles = defaultdict(lambda : defaultdict(set))
        img_verb_role_num = defaultdict(int)
        for image_id in os.listdir(image_dir):
            if image_id not in self.annotation:
                continue
            verb = self.annotation[image_id]['verb']
            if verb not in self.sr_mapping_verb:
                continue
            img_verbs[image_id] = verb.lower()
            # ontology
            # verb_roles[image_id] = self.verbs[verb]['order']
            # role values
            frames = self.annotation[image_id]['frames']
            for frame in frames:
                for role in frame:
                    role = role.lower()
                    role_value_id = frame[role]
                    if len(role_value_id) > 0:
                        role_value = self.nouns[role_value_id]['gloss']
                        # lower()
                        img_verb_roles[image_id][role].update(role_value)
                        # img_verb_role_num[image_id] = img_verb_role_num[image_id] + 1

        return img_verbs, img_verb_roles #, img_verb_role_num

    # idx -> img
    def __getids__(self):
        idx_img = {}
        index = 0
        for img in self.img_verbs:
            idx_img[index] = img
            index += 1

        print("number of images: ", len(idx_img))
        return idx_img

    def __getitem__(self, index):
        """Returns one data pair (image, captions, regions and entities)."""
        img_id = self.idx_imgs[index]
        return self.get_img_info(img_id)


    def get_img_info(self, img_id):
        verb = self.map_to_id(self.img_verbs[img_id], self.vocab_situation_verb.word2id)
        verb_roles = self.img_verb_roles[img_id]  # role -> role_values
        roles = []
        args = []
        for role in verb_roles:
            for role_arg in verb_roles[role]:
                # roles.append(self.make_one_hot(role, self.vocab_situation_role.word2id))
                roles.append(self.map_to_id(role, self.vocab_situation_role.word2id))
                args.append(self.map_to_id(role_arg, self.vocab_situation_noun.word2id))
        arg_num = len(args)

        # roles = np.asarray(roles)

        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image_vec = self.transform(image)

        # load object detection result
        bbox_entities_id = []
        bbox_entities_region = []
        bbox_entities_label = []
        objects = self.object_results[img_id]
        for object in objects:
            label = self.object_label[object['label']]
            bbox = object['bbox']
            score = object['score']
            if score < self.object_detection_threshold:
                continue
            # transform patch to patch_vec
            try:
                patch = image.crop(bbox)
                patch_id = '%d_%d_%d_%d' % (bbox[0], bbox[1], bbox[2], bbox[3])
                bbox_entities_id.append(patch_id)
                bbox_entities_label.append(self.map_to_id(label, self.vocab_situation_noun.word2id))
                if self.transform is not None:
                    patch_vec = self.transform(patch)
                    # bbox_id, bbox_vec, label
                    # bbox_entities.append( (patch_id, patch_vec, label) )
                    bbox_entities_region.append(patch_vec)
                else:
                    # bbox_entities.append( (patch_id, patch, label) )
                    bbox_entities_region.append(patch)
            except:
                print('Wrong image ', img_path)
                traceback.print_exc()
        object_num = len(bbox_entities_region)

        if self.transform is not None:
            return img_id, image_vec, verb, roles, args, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num, object_num
        else:
            return img_id, image, verb, roles, args, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num, object_num

    def __len__(self):
        return len(self.ids)

    # def make_one_hots(self, roles, vocab):
    #     role_vec = np.zeros(len(roles), len(vocab))
    #     for i, r in enumerate(roles):
    #         print(i, r, vocab[r])
    #         role_vec[i][vocab[r]] = 1
    #         # ids = [vocab[t] if t in vocab else consts.UNK_IDX ]
    #     return role_vec

    def make_one_hot(self, role, vocab):
        role_vec = [0] * len(vocab) #np.zeros(len(vocab))
        # print(role)
        # if role not in vocab:
        # print('vocab', vocab)
        role_vec[vocab[role]] = 1
        return role_vec

    def map_to_ids(self, tokens, vocab):
        ids = [vocab[t] if t in vocab else consts.UNK_IDX for t in tokens]
        return ids

    def map_to_id(self, token, vocab):
        if token in vocab:
            return vocab[token]
        else:
            return consts.UNK_IDX


def imsitu_loader(image_dir, vocab_situation_noun, vocab_situation_role, vocab_situation_verb,
                  imsitu_ontology_file, imsitu_annotation_file,
                  verb_mapping_file, role_mapping_file,
                  object_ontology_file, object_detection_pkl_file,
                  object_detection_threshold, transform, batch_size, shuffle=True, num_workers=2):

    """Returns torch.utils.data.DataLoader for custom visualgenome dataset."""
    data_set = ImSituDataset(image_dir=image_dir,
                             vocab_situation_noun=vocab_situation_noun,
                             vocab_situation_role=vocab_situation_role,
                             vocab_situation_verb=vocab_situation_verb,
                             imsitu_ontology_file=imsitu_ontology_file,
                             imsitu_annotation_file=imsitu_annotation_file,
                             verb_mapping_file=verb_mapping_file,
                             role_mapping_file=role_mapping_file,
                             object_ontology_file=object_ontology_file,
                             object_detection_pkl_file=object_detection_pkl_file,
                             object_detection_threshold=object_detection_threshold,
                             transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=image_collate_fn)
    return data_loader

def image_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (img_id, captionvec, regions_tensor, entity_tensor). <-- (__getitem__ !!!)

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption/regions/entities (including padding) is not supported in default.
    """
    # Sort a data list by object_num (descending order).
    batch.sort(key=lambda x: x[-1], reverse=True)
    img_id_batch, image_batch, verb_batch, roles_batch, args_batch, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num_batch, object_num_batch = zip(*batch)  # zip(['a', 'b', 'c'], [1, 2, 3]) =

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_batch = torch.stack(image_batch, 0)

    # object mask

    return img_id_batch, image_batch, verb_batch, roles_batch, args_batch, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num_batch, object_num_batch

def unpack(batch, device):
    img_id_batch = batch[0]
    # print('img_id_batch', img_id_batch)
    # print('device', device)
    image_batch = batch[1].to(device)
    verb_gt_batch = torch.LongTensor(batch[2]).to(device)

    bbox_entities_id = batch[5]
    arg_num_batch = np.array(batch[-2])
    object_num_batch = np.array(batch[-1])

    # roles_gt_batch = torch.LongTensor(batch[3]).to(device)
    # args_batch = torch.LongTensor(batch[4]).to(device)
    arg_num_max = max(arg_num_batch)
    # roles_gt_batch = torch.zeros([len(img_id_batch), arg_num_max, len(batch[3][0][0])]).type(torch.LongTensor).to(device)
    roles_gt_batch = torch.zeros([len(img_id_batch), arg_num_max]).type(torch.LongTensor).to(device)
    args_gt_batch = torch.zeros([len(img_id_batch), arg_num_max]).type(torch.LongTensor).to(device)
    for batch_idx, _ in enumerate(batch[3]):
        roles_gt_batch[batch_idx][:arg_num_batch[batch_idx]] = torch.LongTensor(batch[3][batch_idx])
        args_gt_batch[batch_idx][:arg_num_batch[batch_idx]] = torch.LongTensor(batch[4][batch_idx])
        # for role_idx, _ in enumerate(batch[3][batch_idx]):
        #     roles_gt_batch[batch_idx][role_idx] = torch.LongTensor(batch[3][batch_idx][role_idx])

    object_num_max = max(object_num_batch)
    # bbox_entities_region = torch.zeros(len(img_id_batch), object_num_max, batch[6][0][0].size(0),
    #                                    batch[6][0][0].size(1), batch[6][0][0].size(2)).to(device)
    bbox_entities_region = torch.zeros(len(img_id_batch), object_num_max, 3, 224, 224).to(device)
    bbox_entities_label = torch.zeros(len(img_id_batch), object_num_max).to(device)
    for b_idx, _ in enumerate(batch[6]):
        for obj_idx, _ in enumerate(batch[6][b_idx]):
            bbox_entities_region[b_idx][obj_idx] = batch[6][b_idx][obj_idx]
            bbox_entities_label[b_idx][obj_idx] = batch[7][b_idx][obj_idx]
        # print('b_idx', b_idx)
    return img_id_batch, image_batch, verb_gt_batch, roles_gt_batch, args_gt_batch, \
           bbox_entities_id, bbox_entities_region, bbox_entities_label, \
           arg_num_batch, object_num_batch
