import torch
import torch.utils.data as data
import codecs
import json
from PIL import Image
import os
import pickle
import numpy as np

from src.dataflow.numpy.data_loader_situation import ImSituDataset
from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from torchtext.data import Field, Example
from src.dataflow.torch.Corpus import Corpus
from src.dataflow.torch.Sentence import Sentence_grounding, Sentence_m2e2
from src.dataflow.numpy.data_loader_situation import load_img_object, get_labels

class GroundingDataset(Corpus):
    """
    Defines a dataset composed of Examples along with its Fields.
    """

    sort_key = None

    def __init__(self, path, img_dir, fields, amr=False, transform=None,
                 load_object=False,
                 object_ontology_file=None,
                 object_detection_pkl_file=None,
                 object_detection_threshold=0.2,
                 **kwargs):
        self.img_dir = img_dir
        self.transform = transform

        self.load_object = load_object
        if self.load_object:
            self.object_detection_threshold = object_detection_threshold
            self.object_label = get_labels(object_ontology_file)
            self.object_results = pickle.load(open(object_detection_pkl_file, 'rb'))
        else:
            self.object_detection_threshold = 0.2
            self.object_label = None
            self.object_results = None

        super(GroundingDataset, self).__init__(path, fields, amr, **kwargs)


    def get_object_results(self):
        return self.object_results, self.object_label, self.object_detection_threshold

    def parse_example(self, path, fields, amr, **kwargs):
        examples = []

        _file = codecs.open(path, 'r', 'utf-8')
        jl = json.load(_file)
        print(path, len(jl))
        for js in jl:
            ex = self.parse_sentence(js, fields, amr)
            if ex is not None:
                examples.append(ex)

        return examples

    def parse_sentence(self, js, fields, amr):
        IMAGEID = fields["id"]
        SENTID = fields["sentence_id"]
        # IMAGE = fields["image"]
        WORDS = fields["words"]
        POSTAGS = fields["pos-tags"]
        # LEMMAS = fields["lemma"]
        ENTITYLABELS = fields["golden-entity-mentions"]
        if amr:
            colcc = "simple-parsing"
        else:
            colcc = "combined-parsing"
        ADJMATRIX = fields[colcc]
        ENTITIES = fields["all-entities"]

        sentence = Sentence_grounding(json_content=js, graph_field_name=colcc,
                                      img_dir=self.img_dir, transform=self.transform)
        # if sentence.image_vec is None:
        #     return None
        ex = Example()
        setattr(ex, IMAGEID[0], IMAGEID[1].preprocess(sentence.image_id))
        setattr(ex, SENTID[0], SENTID[1].preprocess(sentence.sentence_id))
        setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.wordList))
        setattr(ex, POSTAGS[0], POSTAGS[1].preprocess(sentence.posLabelList))
        # setattr(ex, LEMMAS[0], LEMMAS[1].preprocess(sentence.lemmaList))
        setattr(ex, ENTITYLABELS[0], ENTITYLABELS[1].preprocess(sentence.entityLabelList))
        setattr(ex, ADJMATRIX[0], (sentence.adjpos, sentence.adjv))
        setattr(ex, ENTITIES[0], ENTITIES[1].preprocess(sentence.entities))
        # setattr(ex, IMAGE[0], IMAGE[1].preprocess(sentence.image_vec))

        return ex

    def longest(self):
        return max([len(x.POSTAGS) for x in self.examples])

def unpack_grounding(batch, device, transform, img_dir_grounding, ee_hyps,
                     load_object=False, object_results=None, object_label=None,
                     object_detection_threshold=.2, vocab_objlabel=None
                     ):
    words, x_len = batch.WORDS
    postags = batch.POSTAGS
    entitylabels = batch.ENTITYLABELS  # entitymention
    adjm = batch.ADJM
    entities = batch.ENTITIES
    image_id = batch.IMAGEID
    sent_id = batch.SENTID

    BATCH_SIZE = words.size()[0]
    SEQ_LEN = words.size()[1]
    adjm = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                 torch.FloatTensor(adjmm[1]),
                                                 torch.Size([ee_hyps["gcn_et"], SEQ_LEN, SEQ_LEN])).to_dense() for
                        adjmm in adjm])
    # image = batch.IMAGE
    image_vec_all = []
    if load_object:
        bbox_entities_id_all = []
        bbox_entities_region_all = []
        bbox_entities_label_all = []
        object_num_all = []

    # for inst in object_results:
    #     print(object_results)

    for img_id in image_id:
        if len(img_id) > 0:
            image, image_vec, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num \
                = load_img_object(img_id, img_dir_grounding, transform,
                                  load_object, object_results, object_label,
                                  object_detection_threshold,
                                  vocab_objlabel)

            if object_num == 0:
                # continue
                return None

            image_vec_all.append(image_vec)
            if load_object:
                bbox_entities_id_all.append(bbox_entities_id)
                bbox_entities_region_all.append(bbox_entities_region)
                bbox_entities_label_all.append(bbox_entities_label)
                object_num_all.append(object_num)  # object_num_batch: [img_num, ]

    # if len(image_vec_all) == 0:
    #     return None, None, None, None, None, \
    #            None, None, \
    #            None, None, None, None, \
    #            None, None

    image = torch.stack(image_vec_all)
    if load_object:
        object_num_final = np.array(object_num_all)
        object_num_max = np.amax(object_num_final, keepdims=False)  # print(object_num_all)
        bbox_entities_region_final = torch.zeros(BATCH_SIZE, object_num_max, 3, 224, 224).to(device)
        bbox_entities_label_final = torch.zeros(BATCH_SIZE, object_num_max).to(device)
        for b_idx, _ in enumerate(bbox_entities_region_all):
            for obj_idx, _ in enumerate(bbox_entities_region_all[b_idx]):
                bbox_entities_region_final[b_idx][obj_idx] = bbox_entities_region_all[b_idx][obj_idx]
                bbox_entities_label_final[b_idx][obj_idx] = bbox_entities_label_all[b_idx][obj_idx]

    words = words.to(device)
    # lemmas = lemmas.to(device)
    x_len = x_len.cpu().detach().numpy()
    postags = postags.to(device)
    adjm = adjm.to(device)
    image = image.to(device)

    if load_object:
        return words, x_len, postags, entitylabels, adjm, \
               image_id, image, \
               bbox_entities_id_all, bbox_entities_region_final, bbox_entities_label_final, object_num_final, \
               sent_id, entities
    else:
        return words, x_len, postags, entitylabels, adjm, \
               image_id, image, \
               None, None, None, None, \
               sent_id, entities


class M2E2Dataset(Corpus):

    sort_key = None

    def __init__(self, path, img_dir, fields, amr=False, transform=None,
                 load_object=False,
                 object_ontology_file=None,
                 object_detection_pkl_file=None,
                 object_detection_threshold=0.2,
                 keep_events=None, only_keep=False,
                 with_sentid=False,
                 **kwargs):
        self.img_dir = img_dir
        self.transform = transform

        self.load_object = load_object
        if self.load_object:
            self.object_detection_threshold = object_detection_threshold
            self.object_label = get_labels(object_ontology_file)
            self.object_results = pickle.load(open(object_detection_pkl_file, 'rb'))
        else:
            self.object_detection_threshold = 0.2
            self.object_label = None
            self.object_results = None

        self.keep_events = keep_events
        self.only_keep = only_keep

        self.with_sentid = with_sentid

        super(M2E2Dataset, self).__init__(path, fields, amr, **kwargs)

    def get_object_results(self):
        return self.object_results, self.object_label, self.object_detection_threshold

    def parse_example(self, path, fields, amr, **kwargs):
        examples = []

        _file = codecs.open(path, 'r', 'utf-8')
        jl = json.load(_file)
        print(path, len(jl))
        for js in jl:
            ex = self.parse_sentence(js, fields, amr)
            if ex is not None:
                examples.append(ex)

        return examples

    def parse_sentence(self, js, fields, amr):
        IMAGEID = fields["image"]
        SENTID = fields["sentence_id"]
        # IMAGE = fields["image"]
        WORDS = fields["words"]
        POSTAGS = fields["pos-tags"]
        ENTITYLABELS = fields["golden-entity-mentions"]
        if amr:
            colcc = "simple-parsing"
        else:
            colcc = "combined-parsing"
        ADJMATRIX = fields[colcc]
        ENTITIES = fields["all-entities"]
        LABELS = fields["golden-event-mentions"]
        EVENTS = fields["all-events"]

        sentence = Sentence_m2e2(json_content=js, graph_field_name=colcc, with_sentid=self.with_sentid)
        # print('sentence.image_id', sentence.image_id)
        # if sentence.image_vec is None:
        #     return None
        ex = Example()
        setattr(ex, IMAGEID[0], IMAGEID[1].preprocess(sentence.image_id))
        setattr(ex, SENTID[0], SENTID[1].preprocess(sentence.sentence_id))
        setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.wordList))
        setattr(ex, POSTAGS[0], POSTAGS[1].preprocess(sentence.posLabelList))
        setattr(ex, ENTITYLABELS[0], ENTITYLABELS[1].preprocess(sentence.entityLabelList))
        setattr(ex, ADJMATRIX[0], (sentence.adjpos, sentence.adjv))
        setattr(ex, ENTITIES[0], ENTITIES[1].preprocess(sentence.entities))
        # setattr(ex, IMAGE[0], IMAGE[1].preprocess(sentence.image_vec))
        setattr(ex, LABELS[0], LABELS[1].preprocess(sentence.triggerLabelList))
        setattr(ex, EVENTS[0], EVENTS[1].preprocess(sentence.events))

        # return ex
        if self.keep_events is not None:
            if self.only_keep and sentence.containsEvents != self.keep_events:
                return None
            elif not self.only_keep and sentence.containsEvents < self.keep_events:
                return None
            else:
                return ex
        else:
            return ex

    def longest(self):
        return max([len(x.POSTAGS) for x in self.examples])


def unpack_m2e2(batch, device, transform, img_dir_m2e2, ee_hyps,
                load_object=False, object_results=None, object_label=None,
                object_detection_threshold=.2, vocab_objlabel=None,
                object_max=10):
    words, x_len = batch.WORDS
    postags = batch.POSTAGS
    entitylabels = batch.ENTITYLABELS  # entitymention
    adjm = batch.ADJM
    entities = batch.ENTITIES
    image_id = batch.IMAGEID
    sent_id = batch.SENTID
    y_gt = batch.LABEL  # event type
    events_gt = batch.EVENT  #args

    BATCH_SIZE = words.size()[0]
    SEQ_LEN = words.size()[1]
    if ee_hyps is None:
        adjm = None
    else:
        adjm = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                     torch.FloatTensor(adjmm[1]),
                                                     torch.Size([ee_hyps["gcn_et"], SEQ_LEN, SEQ_LEN])).to_dense() for
                            adjmm in adjm])
        adjm = adjm.to(device)

    # image = batch.IMAGE
    image_vec_all = []
    if load_object:
        bbox_entities_id_all = []
        bbox_entities_region_all = []
        bbox_entities_label_all = []
        object_num_all = []
        image_num_all = []
    # print('image_id', image_id)
    for img_id_each_batch in image_id:
        image_vec_batch = []
        if load_object:
            bbox_entities_id_batch = []
            bbox_entities_region_batch = []
            bbox_entities_label_batch = []
            object_num_batch = []

        for img_id in img_id_each_batch:
            # print('img_id', img_id)
            if len(img_id) > 0:
                # image_vec_batch.append(transform(Image.open(os.path.join(img_dir_m2e2, img_id)).convert('RGB')))
                image, image_vec, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num \
                    = load_img_object(img_id, img_dir_m2e2, transform,
                                      load_object, object_results, object_label,
                                      object_detection_threshold,
                                      vocab_objlabel)
                image_vec_batch.append(image_vec)
                if load_object:
                    bbox_entities_id_batch.append(bbox_entities_id)
                    bbox_entities_region_batch.append(bbox_entities_region)
                    bbox_entities_label_batch.append(bbox_entities_label)
                    object_num_batch.append(object_num)  # object_num_batch: [img_num, ]

        image_vec_batch = torch.stack(image_vec_batch)
        image_vec_all.append(image_vec_batch)
        if load_object:
            bbox_entities_id_all.append(bbox_entities_id_batch)
            bbox_entities_region_all.append(bbox_entities_region_batch)
            bbox_entities_label_all.append(bbox_entities_label_batch)
            object_num_all.append(object_num_batch)  # object_num_all: [batch, img_num]
            image_num_all.append(len(object_num_batch))
    image = torch.stack(image_vec_all)
    if load_object:
        object_num_final = np.array(object_num_all)
        object_num_max = np.amax(object_num_final, keepdims=False)  # print(object_num_all)
        image_num_final = np.array(image_num_all)
        image_num_max = np.amax(image_num_final, keepdims=False)
        bbox_entities_region_final = torch.zeros(BATCH_SIZE, image_num_max, object_num_max, 3, 224, 224).to(device)
        bbox_entities_label_final = torch.zeros(BATCH_SIZE, image_num_max, object_num_max).to(device)
        for i_idx, _ in enumerate(bbox_entities_region_all):
            for b_idx, _ in enumerate(bbox_entities_region_all[i_idx]):
                for obj_idx, _ in enumerate(bbox_entities_region_all[i_idx][b_idx]):
                    bbox_entities_region_final[i_idx][b_idx][obj_idx] = bbox_entities_region_all[i_idx][b_idx][obj_idx]
                    bbox_entities_label_final[i_idx][b_idx][obj_idx] = bbox_entities_label_all[i_idx][b_idx][obj_idx]

    words = words.to(device)
    # lemmas = lemmas.to(device)
    x_len = x_len.cpu().detach().numpy()
    postags = postags.to(device)
    image = image.to(device)
    y_gt = y_gt.to(device)

    # get image gt annotation


    if load_object:
        return words, x_len, postags, entitylabels, adjm, \
               image_id, image, \
               bbox_entities_id_all, bbox_entities_region_final, bbox_entities_label_final, object_num_final, \
               sent_id, entities, y_gt, events_gt
    else:
        return words, x_len, postags, entitylabels, adjm, \
               image_id, image, \
               None, None, None, None, \
               sent_id, entities, y_gt, events_gt