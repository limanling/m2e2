import argparse
import os
import pickle
import sys
from functools import partial
# import json
import ujson as json
from collections import defaultdict
import shutil

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchtext.data import BucketIterator

import sys
sys.path.append('../..')
from src.util import consts
from torchtext.data import Field
from src.util.vocab import Vocab
from torchtext.vocab import Vectors
from src.dataflow.numpy.data_loader_grounding import GroundingDataset, M2E2Dataset
from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from src.dataflow.numpy.data_loader_situation import ImSituDataset
from src.models.grounding import GroundingModel
from src.eval.EEtesting import EDTester
from src.eval.SRtesting import SRTester
from src.eval.Groundingtesting import GroundingTester
from src.eval.JOINTtesting import JointTester
from src.eval.EEvisualizing import EDVisualizer
from src.engine.SRrunner import load_sr_model
from src.engine.EErunner import load_ee_model, event_role_mask
from src.util.util_model import log
from src.dataflow.numpy.data_loader_grounding import unpack_grounding, unpack_m2e2
from src.models.ace_classifier import ACEClassifier


class JointRunnerTest(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        # keep all training dataset loading, because vacab size and other parameters may be related to it
        # parser.add_argument("--test_voa_text", help="VOA multimedia testing set")
        parser.add_argument("--test_voa_image", help="VOA multimedia testing image dir")
        parser.add_argument("--gt_voa_image", help="VOA multimedia testing image set")
        parser.add_argument("--gt_voa_text", help="VOA multimedia testing news article set")
        parser.add_argument("--gt_voa_align", help="VOA multimedia testing coreference set")
        parser.add_argument("--visual_voa_ee_path", help="VOA multimedia testing text event visualization path")
        parser.add_argument("--visual_voa_ee_gt_ann", help="VOA multimedia testing text event visualization path")
        parser.add_argument("--visual_voa_sr_path", help="VOA multimedia testing image event visualization path")
        parser.add_argument("--visual_voa_g_path", help="VOA multimedia testing grounding visualization path")
        parser.add_argument('--visual_ee_entityfromargs', help="visual_ee_entityfromargs", action='store_true')
        parser.add_argument("--ignore_place_sr_test", help="testing ignore place in sr model", action='store_true')
        parser.add_argument("--ignore_time_test", help="testing ignore place in sr model", action='store_true')
        parser.add_argument('--apply_ee_role_mask', help="apply_ee_role_mask", action='store_true')
        # parser.add_argument('--score_ee', help="score_ee", action='store_true')
        # parser.add_argument('--score_sr', help="score_sr", action='store_true')
        # parser.add_argument('--score_corefer', help="score_corefer", action='store_true')
        parser.add_argument('--keep_events', help="keep the sentence having events larger than int(keep_events)",
                            default=0, type=int)
        parser.add_argument('--keep_events_sr',
                            help="image side keep the sentence having events larger than int(keep_events)",
                            default=0, type=int)
        parser.add_argument('--with_sentid', help="the key of triggers & args with_sentid", action='store_true')
        parser.add_argument('--joint_infer', help="whether use joint inference", action='store_true')

        # grounding  (still need ee, because of the vocab)
        parser.add_argument("--test_grounding", help="grounding validation set")
        parser.add_argument("--train_grounding", help="grounding training set", required=False)
        parser.add_argument("--dev_grounding", help="grounding development set", required=False)
        parser.add_argument("--img_dir_grounding", help="grounding images directory", required=False)
        parser.add_argument("--object_detection_pkl_file_g", help="grounding object_detection_pkl_file", required=False)

        # sr model parameter
        parser.add_argument("--test_sr", help="situation recognition validation set")
        parser.add_argument("--train_sr", help="situation recognition training set", required=False)
        parser.add_argument("--dev_sr", help="situation recognition development set", required=False)
        parser.add_argument("--wnebd", help="situation recognition noun word embedding", required=False)
        parser.add_argument("--wvebd", help="situation recognition verb word embedding", required=False)
        parser.add_argument("--wrebd", help="situation recognition role word embedding", required=False)
        parser.add_argument("--vocab", help="situation recognition vocab_dir", required=False)
        parser.add_argument("--image_dir", help="situation recognition image_dir", required=False)
        parser.add_argument("--imsitu_ontology_file", help="imsitu_ontology_file", required=False)
        parser.add_argument("--object_class_map_file", help="object_class_map_file", required=False)
        parser.add_argument("--object_detection_pkl_file", help="object_detection_pkl_file", required=False)
        parser.add_argument("--object_detection_threshold", default=0.2, type=float, help="object_detection_threshold",
                            required=False)
        parser.add_argument("--verb_mapping_file", help="verb_mapping_file", required=False)
        parser.add_argument("--add_object", help="add_object", action='store_true')
        parser.add_argument("--finetune_sr", help="pretrained sr model path")
        parser.add_argument("--sr_hps", help="situation recognition model hyperparams", required=False)
        parser.add_argument("--sr_hps_path", help="model hyperparams", required=False)
        parser.add_argument("--filter_irrelevant_verbs", help="filter_irrelevant_verbs", action='store_true')
        parser.add_argument("--filter_place", help="filter_place", action='store_true')

        # ee model parameter (still need ee, because of the vocab)
        parser.add_argument("--test_ee", help="event extraction validation set")
        parser.add_argument("--train_ee", help="event extraction training set", required=False)
        parser.add_argument("--dev_ee", help="event extraction development set", required=False)
        parser.add_argument("--webd", help="event extraction word embedding", required=False)
        parser.add_argument("--amr", help="use amr", action='store_true')
        parser.add_argument("--finetune_ee", help="pretrained ee model path")
        parser.add_argument("--ee_hps", help="ee model hyperparams", required=False)
        parser.add_argument("--ee_hps_path", help="model hyperparams", required=False)

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        # parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        # parser.add_argument("--optimizer", default="adam")
        # parser.add_argument("--lr", default=1e-3, type=float)
        # parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        # parser.add_argument("--earlystop", default=999999, type=int)
        # parser.add_argument("--restart", default=999999, type=int)
        # parser.add_argument("--shuffle", help="shuffle", action='store_true')

        parser.add_argument("--device", default="cpu")

        self.a = parser.parse_args()

        if self.a.sr_hps_path:
            self.a.sr_hps = json.load(open(self.a.sr_hps_path))
        if self.a.ee_hps_path:
            self.a.ee_hps = json.load(open(self.a.ee_hps_path))

    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True

        ####################    loading event extraction dataset   ####################
        if self.a.test_ee:
            log('testing event extraction corpus from %s' % self.a.test_ee)
        if self.a.test_ee:
            log('testing event extraction corpus from %s' % self.a.test_ee)

        # both for grounding and ee
        WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        PosTagsField = Field(lower=True, batch_first=True)
        EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
        AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)
        # only for ee
        LabelField = Field(lower=False, batch_first=True, pad_token='0', unk_token=None)
        EventsField = EventField(lower=False, batch_first=True)
        SENTIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)

        if self.a.amr:
            colcc = 'simple-parsing'
        else:
            colcc = 'combined-parsing'
        print(colcc)

        train_ee_set = ACE2005Dataset(path=self.a.train_ee,
                                   fields={"sentence_id": ("SENTID", SENTIDField), "words": ("WORDS", WordsField),
                                           "pos-tags": ("POSTAGS", PosTagsField),
                                           "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                           colcc: ("ADJM", AdjMatrixField),
                                           "golden-event-mentions": ("LABEL", LabelField),
                                           "all-events": ("EVENT", EventsField),
                                           "all-entities": ("ENTITIES", EntitiesField)},
                                   amr=self.a.amr, keep_events=1)

        dev_ee_set = ACE2005Dataset(path=self.a.dev_ee,
                                 fields={"sentence_id": ("SENTID", SENTIDField), "words": ("WORDS", WordsField),
                                         "pos-tags": ("POSTAGS", PosTagsField),
                                         "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                         colcc: ("ADJM", AdjMatrixField),
                                         "golden-event-mentions": ("LABEL", LabelField),
                                         "all-events": ("EVENT", EventsField),
                                         "all-entities": ("ENTITIES", EntitiesField)},
                                 amr=self.a.amr, keep_events=0)

        test_ee_set = ACE2005Dataset(path=self.a.test_ee,
                                  fields={"sentence_id": ("SENTID", SENTIDField), "words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          colcc: ("ADJM", AdjMatrixField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  amr=self.a.amr, keep_events=0)

        print('self.a.train_ee', self.a.train_ee)
        LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL)
        print('LabelField.vocab.stoi', LabelField.vocab.stoi)
        EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT)
        print('EventsField.vocab.stoi', EventsField.vocab.stoi)
        print('len(EventsField.vocab.itos)', len(EventsField.vocab.itos))
        print('len(EventsField.vocab.stoi)', len(EventsField.vocab.stoi))

        ####################    loading SR dataset   ####################
        # both for grounding and sr
        if self.a.train_sr:
            log('loading corpus from %s' % self.a.train_sr)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        vocab_noun = Vocab(os.path.join(self.a.vocab, 'vocab_situation_noun.pkl'), load=True)
        vocab_role = Vocab(os.path.join(self.a.vocab, 'vocab_situation_role.pkl'), load=True)
        vocab_verb = Vocab(os.path.join(self.a.vocab, 'vocab_situation_verb.pkl'), load=True)

        # only need get_role_mask() and sr_mapping()
        train_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
                                     EventsField.vocab.stoi, LabelField.vocab.stoi,
                                     self.a.imsitu_ontology_file,
                                     self.a.train_sr, self.a.verb_mapping_file,
                                     None, None,
                                     0,
                                     transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                     load_object=False, filter_place=self.a.filter_place)


        ####################    loading grounding dataset   ####################
        if self.a.train_grounding:
            log('loading grounding corpus from %s' % self.a.train_grounding)

        # only for grounding
        IMAGEIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        SENTIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        # IMAGEField = SparseField(sequential=False, use_vocab=False, batch_first=True)

        train_grounding_set = GroundingDataset(path=self.a.train_grounding,
                                               img_dir=self.a.img_dir_grounding,
                                               fields={"id": ("IMAGEID", IMAGEIDField),
                                                       "sentence_id": ("SENTID", SENTIDField),
                                                       "words": ("WORDS", WordsField),
                                                       "pos-tags": ("POSTAGS", PosTagsField),
                                                       "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                                       colcc: ("ADJM", AdjMatrixField),
                                                       "all-entities": ("ENTITIES", EntitiesField),
                                                       # "image": ("IMAGE", IMAGEField),
                                                       },
                                               transform=transform,
                                               amr=self.a.amr,
                                               load_object=self.a.add_object,
                                               object_ontology_file=self.a.object_class_map_file,
                                               object_detection_pkl_file=self.a.object_detection_pkl_file_g,
                                               object_detection_threshold=self.a.object_detection_threshold,
                                               )

        dev_grounding_set = GroundingDataset(path=self.a.dev_grounding,
                                             img_dir=self.a.img_dir_grounding,
                                             fields={"id": ("IMAGEID", IMAGEIDField),
                                                     "sentence_id": ("SENTID", SENTIDField),
                                                     "words": ("WORDS", WordsField),
                                                     "pos-tags": ("POSTAGS", PosTagsField),
                                                     "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                                     colcc: ("ADJM", AdjMatrixField),
                                                     "all-entities": ("ENTITIES", EntitiesField),
                                                     # "image": ("IMAGE", IMAGEField),
                                                     },
                                             transform=transform,
                                             amr=self.a.amr,
                                             load_object=self.a.add_object,
                                             object_ontology_file=self.a.object_class_map_file,
                                             object_detection_pkl_file=self.a.object_detection_pkl_file_g,
                                             object_detection_threshold=self.a.object_detection_threshold,
                                             )

        test_grounding_set = GroundingDataset(path=self.a.test_grounding,
                                              img_dir=self.a.img_dir_grounding,
                                              fields={"id": ("IMAGEID", IMAGEIDField),
                                                      "sentence_id": ("SENTID", SENTIDField),
                                                      "words": ("WORDS", WordsField),
                                                      "pos-tags": ("POSTAGS", PosTagsField),
                                                      "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                                      colcc: ("ADJM", AdjMatrixField),
                                                      "all-entities": ("ENTITIES", EntitiesField),
                                                      # "image": ("IMAGE", IMAGEField),
                                                      },
                                              transform=transform,
                                              amr=self.a.amr,
                                              load_object=self.a.add_object,
                                              object_ontology_file=self.a.object_class_map_file,
                                              object_detection_pkl_file=self.a.object_detection_pkl_file_g,
                                              object_detection_threshold=self.a.object_detection_threshold,
                                              )

        ####################    build vocabulary   ####################

        if self.a.webd:
            pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
            WordsField.build_vocab(train_ee_set.WORDS, dev_ee_set.WORDS, train_grounding_set.WORDS, dev_grounding_set.WORDS, vectors=pretrained_embedding)
        else:
            WordsField.build_vocab(train_ee_set.WORDS, dev_ee_set.WORDS, train_grounding_set.WORDS, dev_grounding_set.WORDS)
        PosTagsField.build_vocab(train_ee_set.POSTAGS, dev_ee_set.POSTAGS, train_grounding_set.POSTAGS, dev_grounding_set.POSTAGS)
        EntityLabelsField.build_vocab(train_ee_set.ENTITYLABELS, dev_ee_set.ENTITYLABELS,  train_grounding_set.ENTITYLABELS, dev_grounding_set.ENTITYLABELS)

        consts.O_LABEL = LabelField.vocab.stoi[consts.O_LABEL_NAME]
        # print("O label is", consts.O_LABEL)
        consts.ROLE_O_LABEL = EventsField.vocab.stoi[consts.ROLE_O_LABEL_NAME]
        # print("O label for AE is", consts.ROLE_O_LABEL)

        dev_ee_set1 = ACE2005Dataset(path=self.a.dev_ee,
                                  fields={"sentence_id": ("SENTID", SENTIDField), "words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          colcc: ("ADJM", AdjMatrixField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  amr=self.a.amr, keep_events=1, only_keep=True)

        test_ee_set1 = ACE2005Dataset(path=self.a.test_ee,
                                   fields={"sentence_id": ("SENTID", SENTIDField), "words": ("WORDS", WordsField),
                                           "pos-tags": ("POSTAGS", PosTagsField),
                                           "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                           colcc: ("ADJM", AdjMatrixField),
                                           "golden-event-mentions": ("LABEL", LabelField),
                                           "all-events": ("EVENT", EventsField),
                                           "all-entities": ("ENTITIES", EntitiesField)},
                                   amr=self.a.amr, keep_events=1, only_keep=True)
        print("train set length", len(train_ee_set))

        print("dev set length", len(dev_ee_set))
        print("dev set 1/1 length", len(dev_ee_set1))

        print("test set length", len(test_ee_set))
        print("test set 1/1 length", len(test_ee_set1))

        # sr model initialization
        if not self.a.sr_hps_path:
            self.a.sr_hps = eval(self.a.sr_hps)
        embeddingMatrix_noun = torch.FloatTensor(np.load(self.a.wnebd)).to(self.device)
        embeddingMatrix_verb = torch.FloatTensor(np.load(self.a.wvebd)).to(self.device)
        embeddingMatrix_role = torch.FloatTensor(np.load(self.a.wrebd)).to(self.device)
        if "wvemb_size" not in self.a.sr_hps:
            self.a.sr_hps["wvemb_size"] = len(vocab_verb.id2word)
        if "wremb_size" not in self.a.sr_hps:
            self.a.sr_hps["wremb_size"] = len(vocab_role.id2word)
        if "wnemb_size" not in self.a.sr_hps:
            self.a.sr_hps["wnemb_size"] = len(vocab_noun.id2word)
        if "ae_oc" not in self.a.sr_hps:
            self.a.sr_hps["ae_oc"] = len(vocab_role.id2word)

        self.a.ee_label_weight = torch.ones([len(LabelField.vocab.itos)]) * 5
        self.a.ee_label_weight[consts.O_LABEL] = 1.0
        self.a.ee_arg_weight = torch.ones([len(EventsField.vocab.itos)]) * 5
        if not self.a.ee_hps_path:
            self.a.ee_hps = eval(self.a.ee_hps)
        if "wemb_size" not in self.a.ee_hps:
            self.a.ee_hps["wemb_size"] = len(WordsField.vocab.itos)
        if "pemb_size" not in self.a.ee_hps:
            self.a.ee_hps["pemb_size"] = len(PosTagsField.vocab.itos)
        if "psemb_size" not in self.a.ee_hps:
            # self.a.ee_hps["psemb_size"] = max([train_grounding_set.longest(), dev_grounding_set.longest(), test_grounding_set.longest()]) + 2
            self.a.ee_hps["psemb_size"] = max([train_ee_set.longest(), dev_ee_set.longest(), test_ee_set.longest(), train_grounding_set.longest(), dev_grounding_set.longest(), test_grounding_set.longest()]) + 2
        if "eemb_size" not in self.a.ee_hps:
            self.a.ee_hps["eemb_size"] = len(EntityLabelsField.vocab.itos)
        if "oc" not in self.a.ee_hps:
            self.a.ee_hps["oc"] = len(LabelField.vocab.itos)
        if "ae_oc" not in self.a.ee_hps:
            self.a.ee_hps["ae_oc"] = len(EventsField.vocab.itos)
        if "oc" not in self.a.sr_hps:
            self.a.sr_hps["oc"] = len(LabelField.vocab.itos)
        if "ae_oc" not in self.a.sr_hps:
            self.a.sr_hps["ae_oc"] = len(EventsField.vocab.itos)



        ace_classifier = ACEClassifier(2 * self.a.ee_hps["lstm_dim"], self.a.ee_hps["oc"], self.a.ee_hps["ae_oc"],
                                       self.device)

        # if self.a.score_ee:
        if  self.a.finetune_ee:
            log('init ee model from ' + self.a.finetune_ee)
            ee_model = load_ee_model(self.a.ee_hps, self.a.finetune_ee, WordsField.vocab.vectors, self.device, ace_classifier)
            log('ee model loaded, there are %i sets of params' % len(ee_model.parameters_requires_grads()))
        else:
            ee_model = load_ee_model(self.a.ee_hps, None, WordsField.vocab.vectors, self.device, ace_classifier)
            log('ee model created from scratch, there are %i sets of params' % len(ee_model.parameters_requires_grads()))

        # if self.a.score_sr:
        if self.a.finetune_sr:
            log('init sr model from ' + self.a.finetune_sr)
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, self.a.finetune_sr, self.device, ace_classifier, add_object=self.a.add_object)
            log('sr model loaded, there are %i sets of params' % len(sr_model.parameters_requires_grads()))
        else:
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, None, self.device, ace_classifier, add_object=self.a.add_object)
            log('sr model created from scratch, there are %i sets of params' % len(sr_model.parameters_requires_grads()))

        model = GroundingModel(ee_model, sr_model, self.get_device())
        # ee_model = torch.nn.DataParallel(ee_model)
        # sr_model = torch.nn.DataParallel(sr_model)
        # model = torch.nn.DataParallel(model)

        # if self.a.optimizer == "adadelta":
        #     optimizer_constructor = partial(torch.optim.Adadelta, params=model.parameters_requires_grads(),
        #                                     weight_decay=self.a.l2decay)
        # elif self.a.optimizer == "adam":
        #     optimizer_constructor = partial(torch.optim.Adam, params=model.parameters_requires_grads(),
        #                                     weight_decay=self.a.l2decay)
        # else:
        #     optimizer_constructor = partial(torch.optim.SGD, params=model.parameters_requires_grads(),
        #                                     weight_decay=self.a.l2decay,
        #                                     momentum=0.9)

        # log('optimizer in use: %s' % str(self.a.optimizer))

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)
        with open(os.path.join(self.a.out, "word.vec"), "wb") as f:
            pickle.dump(WordsField.vocab, f)
        with open(os.path.join(self.a.out, "pos.vec"), "wb") as f:
            pickle.dump(PosTagsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "entity.vec"), "wb") as f:
            pickle.dump(EntityLabelsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "label.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "role.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.stoi, f)

        log('init complete\n')

        # ee mappings
        self.a.ee_word_i2s = WordsField.vocab.itos
        self.a.ee_label_i2s = LabelField.vocab.itos
        self.a.ee_role_i2s = EventsField.vocab.itos
        print('TestRunnerJOINT self.a.ee_role_i2s: ', self.a.ee_role_i2s)
        self.a.ee_role_mask = None
        if self.a.apply_ee_role_mask:
            self.a.ee_role_mask = event_role_mask(self.a.train_ee, self.a.dev_ee, LabelField.vocab.stoi, EventsField.vocab.stoi, self.device)
        # sr mappings
        self.a.sr_word_i2s = vocab_noun.id2word
        self.a.sr_label_i2s = vocab_verb.id2word  # LabelField.vocab.itos
        self.a.sr_role_i2s = vocab_role.id2word
        self.a.role_masks = train_sr_set.get_role_mask().to_dense().to(self.device)
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        # loading testing data
        # voa_text = self.a.test_voa_text
        voa_image_dir = self.a.test_voa_image
        gt_voa_image = self.a.gt_voa_image
        gt_voa_text = self.a.gt_voa_text
        gt_voa_align =self.a.gt_voa_align

        sr_verb_mapping, sr_role_mapping = train_sr_set.get_sr_mapping()

        test_m2e2_set = M2E2Dataset(path=gt_voa_text,
                                    img_dir=voa_image_dir,
                                    fields={"image": ("IMAGEID", IMAGEIDField),
                                          "sentence_id": ("SENTID", SENTIDField),
                                          "words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          colcc: ("ADJM", AdjMatrixField),
                                          "all-entities": ("ENTITIES", EntitiesField),
                                          # "image": ("IMAGE", IMAGEField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          },
                                    transform=transform,
                                    amr=self.a.amr,
                                    load_object=self.a.add_object,
                                    object_ontology_file=self.a.object_class_map_file,
                                    object_detection_pkl_file=self.a.object_detection_pkl_file,
                                    object_detection_threshold=self.a.object_detection_threshold,
                                    keep_events=self.a.keep_events,
                                    with_sentid=self.a.with_sentid,
                                    )

        object_results, object_label, object_detection_threshold = test_m2e2_set.get_object_results()

        # build batch on cpu
        test_m2e2_iter = BucketIterator(test_m2e2_set, batch_size=1, train=False,
                                   shuffle=False, device=-1,
                                   sort_key=lambda x: len(x.POSTAGS))

        # scores = 0.0
        # now_bad = 0
        # restart_used = 0
        print("\nStarting testing...\n")
        # lr = parser.lr
        # optimizer = optimizer_constructor(lr=lr)

        ee_tester = EDTester(LabelField.vocab.itos, EventsField.vocab.itos, self.a.ignore_time_test)
        sr_tester = SRTester()
        g_tester = GroundingTester()
        j_tester = JointTester(self.a.ignore_place_sr_test, self.a.ignore_time_test)
        if self.a.visual_voa_ee_path is not None:
            ee_visualizer = EDVisualizer(self.a.gt_voa_text)
        else:
            ee_visualizer = None
        image_gt = json.load(open(gt_voa_image))

        all_y = []
        all_y_ = []
        all_events = []
        all_events_ = []

        vision_result = dict()
        if self.a.visual_voa_g_path is not None and not os.path.exists(self.a.visual_voa_g_path):
            os.makedirs(self.a.visual_voa_g_path, exist_ok=True)
        if self.a.visual_voa_ee_path is not None and not os.path.exists(self.a.visual_voa_ee_path):
            os.makedirs(self.a.visual_voa_ee_path, exist_ok=True)
        if self.a.visual_voa_sr_path is not None and not os.path.exists(self.a.visual_voa_sr_path):
            os.makedirs(self.a.visual_voa_sr_path, exist_ok=True)
        # grounding_writer = open(self.a.visual_voa_g_path, 'w')
        with torch.no_grad():
            model.eval()
            for batch in test_m2e2_iter:
                all_y, all_y_, all_events, all_events_ = joint_test_batch(
                    model_g=model,
                    batch_g=batch,
                    device=self.device,
                    transform=transform,
                    img_dir=voa_image_dir,
                    ee_hyps=self.a.ee_hps,
                    ee_word_i2s=self.a.ee_word_i2s,
                    ee_label_i2s=self.a.ee_label_i2s,
                    ee_role_i2s=self.a.ee_role_i2s,
                    ee_tester=ee_tester,
                    ee_visualizer=ee_visualizer,
                    sr_noun_i2s=self.a.sr_word_i2s,
                    sr_verb_i2s=self.a.sr_label_i2s,
                    sr_role_i2s=self.a.sr_role_i2s,
                    sr_tester=sr_tester,
                    role_masks=self.a.role_masks,
                    ee_role_mask=self.a.ee_role_mask,
                    j_tester=j_tester,
                    image_gt=image_gt,
                    verb2type=sr_verb_mapping,
                    role2role=sr_role_mapping,
                    vision_result=vision_result,
                    all_y=all_y,
                    all_y_=all_y_,
                    all_events=all_events,
                    all_events_=all_events_,
                    visual_g_path=self.a.visual_voa_g_path,
                    visual_ee_path=self.a.visual_voa_ee_path,
                    load_object=self.a.add_object,
                    object_results=object_results,
                    object_label=object_label,
                    object_detection_threshold=object_detection_threshold,
                    vocab_objlabel=vocab_noun.word2id,
                    apply_ee_role_mask=self.a.apply_ee_role_mask,
                    visual_ee_entityfromargs=self.a.visual_ee_entityfromargs,
                    joint_infer=self.a.joint_infer,
                )

        print('vision_result size', len(vision_result))

        ep, er, ef = ee_tester.calculate_report(all_y, all_y_, transform=True)
        ap, ar, af = calculate_sets(all_events, all_events_, self.a.ignore_time_test, EventsField.vocab.itos)
        # if self.a.visual_voa_ee_path is not None:
        #     ee_visualizer.rewrite_brat(self.a.visual_voa_ee_path, self.a.visual_voa_ee_gt_ann,
        #                                self.a.visual_ee_entityfromargs)

        print('text ep, er, ef', ep, er, ef)
        print('text ap, ar, af', ap, ar, af)

        evt_p, evt_r, evt_f1, role_scores = j_tester.calculate_report(
            vision_result, voa_image_dir, self.a.visual_voa_sr_path, self.a.add_object)#consts.O_LABEL, consts.ROLE_O_LABEL)

        print('image event ep, er, ef', evt_p, evt_r, evt_f1)
        if not self.a.add_object:
            print('image att_iou ap, ar, af', role_scores['role_att_iou_p'], role_scores['role_att_iou_r'],
                  role_scores['role_att_iou_f1'])
            print('image att_hit ap, ar, af', role_scores['role_att_hit_p'], role_scores['role_att_hit_r'],
                  role_scores['role_att_hit_f1'])
            print('image att_cor ap, ar, af', role_scores['role_att_cor_p'], role_scores['role_att_cor_r'],
                  role_scores['role_att_cor_f1'])
        else:
            print('image obj_iou ap, ar, af', role_scores['role_obj_iou_p'], role_scores['role_obj_iou_r'],
                  role_scores['role_obj_iou_f1'])

        # with open(os.path.join(self.a.visual_voa_sr_path,'image_prediction.json'), 'w') as f:
        #     json.dump(vision_result, f, indent=2)


def calculate_sets(y, y_, ignore_time, role_i2s, with_sentid=False):
    ct, p1, p2 = 0, 0, 0
    for sent, sent_ in zip(y, y_):
        print(sent, sent_)
        # trigger start
        for key, value in sent.items():
            # key = trigger end, event type
            # value = args
            p1 += len(value)
            if key not in sent_:
                continue
            # matched sentences
            arguments = value
            arguments_ = sent_[key]
            for item, item_ in zip(arguments, arguments_):
                if ignore_time and role_i2s[item[2]].upper().startswith('TIME'):
                    continue
                if item[2] == item_[2]:
                    ct += 1

        for key, value in sent_.items():
            # p2 += len(value)
            # print('key', key)
            for item in sent_[key]:
                if ignore_time and role_i2s[item[2]].upper().startswith('TIME'):
                    continue
                p2 += 1


    if ct == 0 or p1 == 0 or p2 == 0:
        return 0.0, 0.0, 0.0
    else:
        p = 1.0 * ct / p2
        r = 1.0 * ct / p1
        f1 = 2.0 * p * r / (p + r)
        return p, r, f1


def joint_test_batch(model_g, batch_g, device, transform, img_dir, ee_hyps,
                     ee_word_i2s, ee_label_i2s, ee_role_i2s, ee_tester, ee_visualizer,
                     sr_noun_i2s, sr_verb_i2s, sr_role_i2s, sr_tester,
                     role_masks, ee_role_mask, j_tester, image_gt, verb2type, role2role, vision_result,
                     all_y, all_y_, all_events, all_events_, visual_g_path, visual_ee_path,
                     load_object=False, object_results=None, object_label=None,
                     object_detection_threshold=.2, vocab_objlabel=None,
                     apply_ee_role_mask=False, visual_ee_entityfromargs=False,
                     joint_infer=False,
                     ):

    # print(verb2type)
    # print(role2role)

    ######################### Joint ######################
    batch_unpacked = unpack_m2e2(batch_g, device, transform, img_dir, ee_hyps,
                                 load_object, object_results, object_label,
                                 object_detection_threshold, vocab_objlabel)
    words, x_len, postags, entitylabels, adjm, image_id, image, \
        bbox_entities_id_all, bbox_entities_region_all, bbox_entities_label_all, object_num_all, \
        sent_id, entities, y_gt, events_gt = batch_unpacked
    # if load_object:
    #     print('image_id', image_id)
    #     print('bbox_entities_id_all', bbox_entities_id_all)
    #     print('bbox_entities_region_all', bbox_entities_region_all.size())
    #     print('object_num_all', object_num_all)

    type_logits, predicted_event_triggers_batch, predicted_events, sr_verb_, sr_role_, heatmap, grounding_score = \
        model_g.predict(words, x_len, postags, entitylabels, adjm, entities, image_id, image,
                        ee_label_i2s, role_masks, ee_role_mask,
                        add_object=load_object, bbox_entities_id=bbox_entities_id_all, bbox_entities_region=bbox_entities_region_all,
                        bbox_entities_label=bbox_entities_label_all, object_num_batch=object_num_all,
                        apply_ee_role_mask=apply_ee_role_mask, sent_id=sent_id, joint_infer=joint_infer)
    all_events_.extend(predicted_events)
    all_events.extend(events_gt)
    type_logits_max, type_logits_idx = torch.max(type_logits, 2)
    y__ = type_logits_idx.tolist()
    y = y_gt.tolist()

    # for st in predicted_event_triggers:
    #     ed, trigger_type_str = predicted_event_triggers[st]
    for i, ll in enumerate(x_len):
        y[i] = y[i][:ll]
        y__[i] = y__[i][:ll]
    all_y.extend(y)
    all_y_.extend(y__)

    # evaluate each image
    image_done = set()
    # evaluate each image with each sentence
    # evaluate image
    for batch_id in range(len(image_id)):
        doc_id = sent_id[batch_id][:sent_id[batch_id].find('.rsd')]


        words_str = [ee_word_i2s[word_idx] for word_idx in words[batch_id]]
        if visual_ee_path is not None:
            # visual_file_path = os.path.join(visual_ee_path, doc_id+'.rsd.ann_tmp')
            # visual_ee_writer = open(visual_file_path, 'a')
            # ee_visualizer.visualize_brat(predicted_event_triggers_batch[batch_id], predicted_events[batch_id],
            #                          sent_id[batch_id], ee_role_i2s, visual_ee_writer, save_entity=visual_ee_entityfromargs)
            visual_file_path = os.path.join(visual_ee_path, doc_id + '.html')
            visual_ee_writer = open(visual_file_path, 'a')
            ee_visualizer.visualize_html(predicted_event_triggers_batch[batch_id], predicted_events[batch_id],
                                     sent_id[batch_id], ee_role_i2s, visual_ee_writer)

        for img_idx in range(len(image_id[batch_id])):
            img_id = image_id[batch_id][img_idx]
            if img_id in image_done:
                continue
            verb_name = sr_verb_i2s[sr_verb_[batch_id][img_idx]]
            if verb_name not in verb2type:
                continue
            ace_type_name = verb2type[verb_name]
            vision_result[img_id] = dict()
            if img_id.replace('.jpg', '') in image_gt:
                vision_result[img_id]['ground_truth'] = image_gt[img_id.replace('.jpg', '')]
            else:
                vision_result[img_id]['ground_truth'] = dict()
                vision_result[img_id]['ground_truth']['role'] = dict()
                vision_result[img_id]['ground_truth']['event_type'] = consts.O_LABEL_NAME
            vision_result[img_id]['verb'] = verb_name
            vision_result[img_id]['event_type'] = ace_type_name
            vision_result[img_id]['role'] = dict()
            if not load_object:
                vision_result[img_id]['heatmap'] = dict()
            else:
                vision_result[img_id]['object'] = dict()
            for role_idx in range(len(sr_role_[batch_id][img_idx])):
                if sr_role_[batch_id][img_idx][role_idx] != 0:
                    sr_role_name = sr_role_i2s[role_idx]
                    ace_role_name = role2role[verb_name][sr_role_name]
                    if not load_object:
                        vision_result[img_id]['role'][ace_role_name] = sr_noun_i2s[
                            sr_role_[batch_id][img_idx][role_idx]]
                        vision_result[img_id]['heatmap'][ace_role_name] = heatmap[batch_id][img_idx][role_idx].detach().cpu().numpy()
                    else:
                        vision_result[img_id]['role'][ace_role_name] = list()
                        obj_num = object_num_all[batch_id][img_idx].item()
                        if role_idx < obj_num:
                            obj_current_id = bbox_entities_id_all[batch_id][img_idx][role_idx]
                            obj_current_bbox = [int(x) for x in bbox_entities_id_all[batch_id][img_idx][role_idx].split('_')]
                            obj_current_region = bbox_entities_region_all[batch_id][img_idx][role_idx]
                            obj_current_label = bbox_entities_label_all[batch_id][img_idx][role_idx]
                            vision_result[img_id]['role'][ace_role_name].append( (obj_current_id, obj_current_bbox,
                                                                   obj_current_region, obj_current_label) )

            image_done.add(img_id)

    if visual_g_path is not None:
        visual_writer = open(os.path.join(visual_g_path, doc_id + '.html'), 'a+')
        visual_writer.write('==============================')
        visual_writer.write('grounding score %s <br>' % grounding_score[batch_id])
        visual_writer.write('sent_id %s <br>' % sent_id[batch_id])
        visual_writer.write('words %s <br>' % str(words_str))
        visual_writer.write('predicted_event_triggers_batch %s <br>' %
                            str(predicted_event_triggers_batch[
                                    batch_id]))  # batchid -> (trigger_start, trigger_end, event type)
        visual_writer.write('predicted_events %s <br>' % str(predicted_events[
                                                                 batch_id]))  # batch_id -> (trigger_start, trigger_end, event type) -> (entity_start, entity_end, entity_type)
        visual_writer.write('ground_truth_events %s <br>' % str(events_gt[batch_id]))
        for event in predicted_events[batch_id]:
            visual_writer.write('event %s <br>' % str(event))
            for role in predicted_events[batch_id][event]:
                visual_writer.write('%s, %s' % (str(words_str[role[0]: role[1]]), str(ee_role_i2s[role[2]])))
        visual_writer.write('images %s <br>' % image_id[batch_id])

        image_done = set()
        for img_idx in range(len(image_id[batch_id])):
            img_id = image_id[batch_id][img_idx]
            if img_id in image_done:
                continue
            visual_writer.write('image_id %s <br>' % img_id)
            visual_writer.write('<img src=\"./VOA_image_en/'+img_id+'\" width=\"400\">\n<br>\n')
            verb_name = sr_verb_i2s[sr_verb_[batch_id][img_idx]]
            if verb_name not in verb2type:
                visual_writer.write('No ACE Event, %s <br>' % verb_name)
                continue
            ace_type_name = verb2type[verb_name]
            visual_writer.write('sr_verb_, %s, %s <br>' % (verb_name, ace_type_name))
            if img_id.replace('.jpg', '') in image_gt:
                visual_writer.write(str(image_gt[img_id.replace('.jpg', '')]))
            for role_idx in range(len(sr_role_[batch_id][img_idx])):
                if sr_role_[batch_id][img_idx][role_idx] != 0:
                    sr_role_name = sr_role_i2s[role_idx]
                    ace_role_name = role2role[verb_name][sr_role_name]
                    visual_writer.write('%s %s = %s <br>' % (ace_role_name, sr_role_name, sr_noun_i2s[sr_role_[batch_id][img_idx][role_idx]]))
            image_done.add(img_id)

        visual_writer.flush()
        visual_writer.close()

    return all_y, all_y_, all_events, all_events_

if __name__ == "__main__":
    JointRunnerTest().run()
