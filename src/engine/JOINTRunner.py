import argparse
import os
import pickle
import json
import sys
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms

import sys
sys.path.append('../..')
from src.util import consts
from torchtext.data import Field
from src.util.vocab import Vocab
from torchtext.vocab import Vectors
from src.dataflow.numpy.data_loader_grounding import GroundingDataset
from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from src.dataflow.numpy.data_loader_situation import ImSituDataset
from src.models.grounding import GroundingModel
from src.eval.EEtesting import EDTester
from src.eval.SRtesting import SRTester
from src.eval.Groundingtesting import GroundingTester
from src.eval.JOINTtesting import JointTester
from src.engine.JOINTTraining_sample import joint_train
from src.engine.SRrunner import load_sr_model
from src.engine.EErunner import load_ee_model
from src.util.util_model import log
from src.engine.EErunner import event_role_mask
from src.models.ace_classifier import ACEClassifier

class JointRunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        # grounding
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
        parser.add_argument("--filter_irrelevant_verbs", help="filter_irrelevant_verbs", action='store_true')
        parser.add_argument("--filter_place", help="filter_place", action='store_true')
        parser.add_argument("--sr_hps_path", help="model hyperparams", required=False)

        # ee model parameter
        parser.add_argument("--test_ee", help="event extraction validation set")
        parser.add_argument("--train_ee", help="event extraction training set", required=False)
        parser.add_argument("--dev_ee", help="event extraction development set", required=False)
        parser.add_argument("--webd", help="event extraction word embedding", required=False)
        parser.add_argument("--amr", help="use amr", action='store_true')
        parser.add_argument("--finetune_ee", help="pretrained ee model path")
        parser.add_argument("--ee_hps", help="ee model hyperparams", required=False)

        parser.add_argument("--ignore_place_sr_test", help="testing ignore place in sr model", action='store_true')
        parser.add_argument("--ignore_time_test", help="testing ignore place in sr model", action='store_true')

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--earlystop", default=999999, type=int)
        parser.add_argument("--restart", default=999999, type=int)
        parser.add_argument("--shuffle", help="shuffle", action='store_true')

        parser.add_argument("--device", default="cpu")

        self.a = parser.parse_args()

        if self.a.sr_hps_path:
            self.a.sr_hps = json.load(open(self.a.sr_hps_path))

    def set_device(self, device="cpu"):
        # self.device = torch.device(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    # def load_model(self, ed_model, sr_model, fine_tune):
    #     if fine_tune is None:
    #         train_model = GroundingModel(ed_model, sr_model, self.get_device())
    #         return train_model
    #     else:
    #         mymodel = GroundingModel(ed_model, sr_model, self.get_device())
    #         mymodel.load_model(fine_tune)
    #         mymodel.to(self.get_device())
    #         return mymodel

    # def get_tester(self):
    #     return GroundingTester()

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True

        ####################    loading event extraction dataset   ####################
        if self.a.train_ee:
            log('loading event extraction corpus from %s' % self.a.train_ee)

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

        if self.a.webd:
            pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
            LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL, vectors=pretrained_embedding)
            EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT, vectors=pretrained_embedding)
        else:
            LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL)
            EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT)

        # add role mask
        self.a.role_mask = event_role_mask(self.a.train_ee, self.a.dev_ee, LabelField.vocab.stoi,
                                           EventsField.vocab.stoi, self.device)

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

        # train_sr_loader = imsitu_loader(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, self.a.imsitu_ontology_file,
        #                             self.a.train_sr, self.a.verb_mapping_file, self.a.role_mapping_file,
        #                             self.a.object_class_map_file, self.a.object_detection_pkl_file,
        #                             self.a.object_detection_threshold,
        #                             transform, self.a.batch, shuffle=self.a.shuffle, num_workers=1)  #self.a.shuffle
        # dev_sr_loader = imsitu_loader(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, self.a.imsitu_ontology_file,
        #                             self.a.dev_sr, self.a.verb_mapping_file, self.a.role_mapping_file,
        #                             self.a.object_class_map_file, self.a.object_detection_pkl_file,
        #                             self.a.object_detection_threshold,
        #                             transform, self.a.batch, shuffle=self.a.shuffle, num_workers=1)
        # test_sr_loader = imsitu_loader(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, self.a.imsitu_ontology_file,
        #                             self.a.test_sr, self.a.verb_mapping_file, self.a.role_mapping_file,
        #                             self.a.object_class_map_file, self.a.object_detection_pkl_file,
        #                             self.a.object_detection_threshold,
        #                             transform, self.a.batch, shuffle=self.a.shuffle, num_workers=1)
        train_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
                                     LabelField.vocab.stoi, EventsField.vocab.stoi,
                                     self.a.imsitu_ontology_file,
                                     self.a.train_sr, self.a.verb_mapping_file,
                                     self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                     self.a.object_detection_threshold,
                                     transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                     load_object=self.a.add_object, filter_place=self.a.filter_place)
        dev_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
                                   LabelField.vocab.stoi, EventsField.vocab.stoi,
                                   self.a.imsitu_ontology_file,
                                   self.a.dev_sr, self.a.verb_mapping_file,
                                   self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                   self.a.object_detection_threshold,
                                   transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                   load_object=self.a.add_object, filter_place=self.a.filter_place)
        test_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
                                    LabelField.vocab.stoi, EventsField.vocab.stoi,
                                    self.a.imsitu_ontology_file,
                                    self.a.test_sr, self.a.verb_mapping_file,
                                    self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                    self.a.object_detection_threshold,
                                    transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                    load_object=self.a.add_object, filter_place=self.a.filter_place)


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

        self.a.ee_label_weight = torch.ones([len(LabelField.vocab.itos)]) * 5
        self.a.ee_label_weight[consts.O_LABEL] = 1.0
        self.a.ee_arg_weight = torch.ones([len(EventsField.vocab.itos)]) * 5
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

        ee_tester = EDTester(LabelField.vocab.itos, EventsField.vocab.itos, self.a.ignore_time_test)
        sr_tester = SRTester()
        g_tester = GroundingTester()
        j_tester = JointTester(self.a.ignore_place_sr_test, self.a.ignore_time_test)

        ace_classifier = ACEClassifier(2 * self.a.ee_hps["lstm_dim"], self.a.ee_hps["oc"], self.a.ee_hps["ae_oc"], self.device)

        if self.a.finetune_ee:
            log('init ee model from ' + self.a.finetune_ee)
            ee_model = load_ee_model(self.a.ee_hps, self.a.finetune_ee, WordsField.vocab.vectors, self.device, ace_classifier)
            log('ee model loaded, there are %i sets of params' % len(ee_model.parameters_requires_grads()))
        else:
            ee_model = load_ee_model(self.a.ee_hps, None, WordsField.vocab.vectors, self.device, ace_classifier)
            log('ee model created from scratch, there are %i sets of params' % len(ee_model.parameters_requires_grads()))

        if self.a.finetune_sr:
            log('init sr model from ' + self.a.finetune_sr)
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, self.a.finetune_sr, self.device, ace_classifier, add_object=self.a.add_object, load_partial=True)
            log('sr model loaded, there are %i sets of params' % len(sr_model.parameters_requires_grads()))
        else:
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, None, self.device, ace_classifier, add_object=self.a.add_object, load_partial=True)
            log('sr model created from scratch, there are %i sets of params' % len(sr_model.parameters_requires_grads()))

        model = GroundingModel(ee_model, sr_model, self.get_device())
        # ee_model = torch.nn.DataParallel(ee_model)
        # sr_model = torch.nn.DataParallel(sr_model)
        # model = torch.nn.DataParallel(model)

        if self.a.optimizer == "adadelta":
            optimizer_constructor = partial(torch.optim.Adadelta, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        elif self.a.optimizer == "adam":
            optimizer_constructor = partial(torch.optim.Adam, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        else:
            optimizer_constructor = partial(torch.optim.SGD, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay,
                                            momentum=0.9)

        log('optimizer in use: %s' % str(self.a.optimizer))

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
        with open(os.path.join(self.a.out, "ee_hyps.json"), "w") as f:
            json.dump(self.a.ee_hps, f)
        with open(os.path.join(self.a.out, "sr_hyps.json"), "w") as f:
            json.dump(self.a.sr_hps, f)

        log('init complete\n')

        # ee mappings
        self.a.ee_word_i2s = WordsField.vocab.itos
        self.a.ee_label_i2s = LabelField.vocab.itos
        self.a.ee_role_i2s = EventsField.vocab.itos
        # sr mappings
        self.a.sr_word_i2s = vocab_noun.id2word
        self.a.sr_label_i2s = vocab_verb.id2word  # LabelField.vocab.itos
        self.a.sr_role_i2s = vocab_role.id2word
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        joint_train(
            model_ee=ee_model,
            model_sr=sr_model,
            model_g=model,
            train_set_g=train_grounding_set,
            dev_set_g=dev_grounding_set,
            test_set_g=test_grounding_set,
            train_set_ee=train_ee_set,
            dev_set_ee=dev_ee_set,
            test_set_ee=test_ee_set,
            train_set_sr=train_sr_set,
            dev_set_sr=dev_sr_set,
            test_set_sr=test_sr_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            ee_tester=ee_tester,
            sr_tester=sr_tester,
            g_tester=g_tester,
            j_tester=j_tester,
            parser=self.a,
            other_testsets={
                "dev ee 1/1": dev_ee_set1,
                "test ee 1/1": test_ee_set1,
            },
            transform=transform,
            vocab_objlabel=vocab_noun.word2id
        )
        log('Done!')


if __name__ == "__main__":
    JointRunner().run()
