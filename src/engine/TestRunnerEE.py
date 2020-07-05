import argparse
import os
import pickle
import json
import sys
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Field
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from math import ceil

import sys
#export PATH=/dvmm-filer2/users/manling/mm-event-graph2:$PATH
sys.path.append('../..')

from src.util import consts
from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from src.models.ee import EDModel
from src.eval.EEtesting import EDTester
from src.engine.EEtraining import ee_train
from src.util.util_model import log
from src.engine.EEtraining import run_over_data
from src.engine.EErunner import load_ee_model
from src.util.util_model import progressbar
from src.dataflow.torch.Sentence import Token
from src.engine.EErunner import event_role_mask


class EERunnerTest(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test_ee", help="event extraction validation set")
        parser.add_argument("--train_ee", help="event extraction training set", required=False)
        parser.add_argument("--dev_ee", help="event extraction development set", required=False)
        parser.add_argument("--webd", help="word embedding", required=False)
        parser.add_argument("--ignore_time_test", help="testing ignore place in sr model", action='store_true')

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        # parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=1111, type=int)
        # parser.add_argument("--optimizer", default="adam")
        # parser.add_argument("--lr", default=1, type=float)
        # parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        # parser.add_argument("--earlystop", default=999999, type=int)
        # parser.add_argument("--restart", default=999999, type=int)
        # parser.add_argument("--shuffle", help="shuffle", action='store_true')
        parser.add_argument("--amr", help="use amr", action='store_true')

        parser.add_argument("--device", default="cpu")
        parser.add_argument("--hps_path", help="model hyperparams", required=False)
        parser.add_argument("--hps", help="model hyperparams", required=False)

        self.a = parser.parse_args()
        if self.a.hps_path:
            self.a.hps = json.load(open(self.a.hps_path))
        print(self.a.hps)

    def set_device(self, device="cpu"):
        # self.device = torch.device(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def get_tester(self, voc_i2s, voc_role_i2s):
        return EDTester(voc_i2s, voc_role_i2s, self.a.ignore_time_test)

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True

        # create training set
        if self.a.test_ee:
            log('loading event extraction corpus from %s' % self.a.test_ee)

        WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        PosTagsField = Field(lower=True, batch_first=True)
        EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
        AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        LabelField = Field(lower=False, batch_first=True, pad_token='0', unk_token=None)
        EventsField = EventField(lower=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)
        if self.a.amr:
            colcc = 'amr-colcc'
        else:
            colcc = 'stanford-colcc'
        print(colcc)

        train_ee_set = ACE2005Dataset(path=self.a.train_ee,
                                      fields={"words": ("WORDS", WordsField),
                                              "pos-tags": ("POSTAGS", PosTagsField),
                                              "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                              colcc: ("ADJM", AdjMatrixField),
                                              "golden-event-mentions": ("LABEL", LabelField),
                                              "all-events": ("EVENT", EventsField),
                                              "all-entities": ("ENTITIES", EntitiesField)},
                                      amr=self.a.amr, keep_events=1)

        dev_ee_set = ACE2005Dataset(path=self.a.dev_ee,
                                    fields={"words": ("WORDS", WordsField),
                                            "pos-tags": ("POSTAGS", PosTagsField),
                                            "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                            colcc: ("ADJM", AdjMatrixField),
                                            "golden-event-mentions": ("LABEL", LabelField),
                                            "all-events": ("EVENT", EventsField),
                                            "all-entities": ("ENTITIES", EntitiesField)},
                                    amr=self.a.amr, keep_events=0)

        test_ee_set = ACE2005Dataset(path=self.a.test_ee,
                                     fields={"words": ("WORDS", WordsField),
                                             "pos-tags": ("POSTAGS", PosTagsField),
                                             "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                             colcc: ("ADJM", AdjMatrixField),
                                             "golden-event-mentions": ("LABEL", LabelField),
                                             "all-events": ("EVENT", EventsField),
                                             "all-entities": ("ENTITIES", EntitiesField)},
                                     amr=self.a.amr, keep_events=0)

        if self.a.webd:
            pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
            WordsField.build_vocab(train_ee_set.WORDS, dev_ee_set.WORDS, vectors=pretrained_embedding)
        else:
            WordsField.build_vocab(train_ee_set.WORDS, dev_ee_set.WORDS)
        PosTagsField.build_vocab(train_ee_set.POSTAGS, dev_ee_set.POSTAGS)
        EntityLabelsField.build_vocab(train_ee_set.ENTITYLABELS, dev_ee_set.ENTITYLABELS)
        LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL)
        EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT)
        consts.O_LABEL = LabelField.vocab.stoi[consts.O_LABEL_NAME]
        # print("O label is", consts.O_LABEL)
        consts.ROLE_O_LABEL = EventsField.vocab.stoi[consts.ROLE_O_LABEL_NAME]
        # print("O label for AE is", consts.ROLE_O_LABEL)

        self.a.label_weight = torch.ones([len(LabelField.vocab.itos)]) * 5
        self.a.label_weight[consts.O_LABEL] = 1.0
        self.a.arg_weight = torch.ones([len(EventsField.vocab.itos)]) * 5
        # add role mask
        self.a.role_mask = event_role_mask(self.a.test_ee, self.a.train_ee, self.a.dev_ee, LabelField.vocab.stoi,
                                           EventsField.vocab.stoi, self.device)
        # print('self.a.hps', self.a.hps)
        if not self.a.hps_path:
            self.a.hps = eval(self.a.hps)
        if "wemb_size" not in self.a.hps:
            self.a.hps["wemb_size"] = len(WordsField.vocab.itos)
        if "pemb_size" not in self.a.hps:
            self.a.hps["pemb_size"] = len(PosTagsField.vocab.itos)
        if "psemb_size" not in self.a.hps:
            self.a.hps["psemb_size"] = max([train_ee_set.longest(), dev_ee_set.longest(), test_ee_set.longest()]) + 2
        if "eemb_size" not in self.a.hps:
            self.a.hps["eemb_size"] = len(EntityLabelsField.vocab.itos)
        if "oc" not in self.a.hps:
            self.a.hps["oc"] = len(LabelField.vocab.itos)
        if "ae_oc" not in self.a.hps:
            self.a.hps["ae_oc"] = len(EventsField.vocab.itos)

        tester = self.get_tester(LabelField.vocab.itos, EventsField.vocab.itos)

        if self.a.finetune:
            log('init model from ' + self.a.finetune)
            model = load_ee_model(self.a.hps, self.a.finetune, WordsField.vocab.vectors, self.device)
            log('model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
        else:
            model = load_ee_model(self.a.hps, None, WordsField.vocab.vectors, self.device)
            log('model created from scratch, there are %i sets of params' % len(model.parameters_requires_grads()))

        self.a.word_i2s = WordsField.vocab.itos
        self.a.label_i2s = LabelField.vocab.itos
        self.a.role_i2s = EventsField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        # train_iter = BucketIterator(train_ee_set, batch_size=self.a.batch,
        #                             train=True, shuffle=False, device=-1,
        #                             sort_key=lambda x: len(x.POSTAGS))
        # dev_iter = BucketIterator(dev_ee_set, batch_size=self.a.batch, train=False,
        #                           shuffle=False, device=-1,
        #                           sort_key=lambda x: len(x.POSTAGS))
        test_iter = BucketIterator(test_ee_set, batch_size=self.a.batch, train=False,
                                   shuffle=False, device=-1,
                                   sort_key=lambda x: len(x.POSTAGS))

        print("\nStarting testing ...\n")

        # Testing Phrase
        test_loss, test_ed_p, test_ed_r, test_ed_f1, \
        test_ae_p, test_ae_r, test_ae_f1 = run_over_data(data_iter=test_iter,
                                                         optimizer=None,
                                                         model=model,
                                                         need_backward=False,
                                                         MAX_STEP=ceil(len(
                                                             test_ee_set) /
                                                                       self.a.batch),
                                                         tester=tester,
                                                         hyps=model.hyperparams,
                                                         device=model.device,
                                                         maxnorm=self.a.maxnorm,
                                                         word_i2s=self.a.word_i2s,
                                                         label_i2s=self.a.label_i2s,
                                                         role_i2s=self.a.role_i2s,
                                                         weight=self.a.label_weight,
                                                         arg_weight=self.a.arg_weight,
                                                         save_output=os.path.join(
                                                             self.a.out,
                                                             "test_final.txt"),
                                                         role_mask=self.a.role_mask)

        print("\nFinally test loss: ", test_loss,
              "\ntest ed p: ", test_ed_p,
              " test ed r: ", test_ed_r,
              " test ed f1: ", test_ed_f1,
              "\ntest ae p: ", test_ae_p,
              " test ae r: ", test_ae_r,
               " test ae f1: ", test_ae_f1)

if __name__ == "__main__":
    EERunnerTest().run()