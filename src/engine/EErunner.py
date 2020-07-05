import argparse
import os
import pickle
import json
import sys
from functools import partial
import json
from collections import defaultdict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Field
from torchtext.vocab import Vectors

import sys
#export PATH=/dvmm-filer2/users/manling/mm-event-graph2:$PATH
sys.path.append('../..')

from src.util import consts
from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from src.models.ee import EDModel
from src.models.ace_classifier import ACEClassifier
from src.eval.EEtesting import EDTester
from src.engine.EEtraining import ee_train
from src.util.util_model import log


def load_ee_model(hps, fine_tune, pretrained_embedding, device, ace_classifier=None):
    assert pretrained_embedding is not None
    if ace_classifier is None:
        ace_classifier = ACEClassifier(2 * hps["lstm_dim"], hps["oc"], hps["ae_oc"], device)
    if fine_tune is None:
        return EDModel(hps, device, pretrained_embedding, ace_classifier)
    else:
        # ace_classifier.load_model(fine_tune_classifier)
        mymodel = EDModel(hps, device, pretrained_embedding, ace_classifier)
        mymodel.load_model(fine_tune)
        mymodel.to(device)
        return mymodel


def event_role_mask(train_ee, dev_ee, type2id, role2id, device):
    # get ontology structure
    ont_dict = defaultdict(lambda: defaultdict(set))
    for file in [train_ee, dev_ee]:
        ee_json = json.load(open(file))
        # entity_type_set = set()  # 'WEA', 'ORG', 'GPE', 'LOC', 'TIME', 'VEH', 'VALUE', 'FAC', 'PER'
        # role_set = defaultdict(int)
        for sent in ee_json:
            entities = sent['golden-entity-mentions']
            entity_dict = defaultdict(lambda: defaultdict(list))
            for entity in entities:
                entity_start = entity['start']
                entity_end = entity['end']
                # if coarse_type:
                entity_type = entity['entity-type'].split(':')[0]
                # else:
                #     entity_type = entity['entity-type']
                # entity_type_set.add(entity_type)
                entity_dict[entity_start][entity_end] = entity_type
            events = sent['golden-event-mentions']
            for event in events:
                event_type = event['event_type'].replace(':', '||').replace('-', '|')
                args = event['arguments']
                for arg in args:
                    # role = '%s_%s' % (event_type, arg['role'])
                    role = arg['role']
                    # role_set.add(role)
                    # role_set[role] += 1
                    arg_start = arg['start']
                    arg_end = arg['end']
                    # print(event_type)
                    # print(role)
                    # print(arg_start)
                    # print(arg_end)
                    # print(entity_dict[arg_start][arg_end])
                    if arg_end not in entity_dict[arg_start]:
                        continue
                    ont_dict[event_type.upper()][role.upper()].add(entity_dict[arg_start][arg_end])

    # get 'OTHER' idx
    ROLE_O_LABEL = role2id[consts.ROLE_O_LABEL_NAME]
    # print("O label for AE is", consts.ROLE_O_LABEL)

    # generate mask
    row_indexes = []
    column_indexes = []
    typestr2id = dict()
    # print(type2id)
    for event_type_long in type2id:
        event_type = event_type_long.replace('I-', '').replace('B-','')
        event_type_id = type2id[event_type_long]
        typestr2id[event_type] = event_type_id
        cnt = 0
        for role_long in role2id:
            role = role_long.replace('I-', '').replace('B-', '')
            if event_type in ont_dict:
                # add 'OTHER'
                column_indexes.append(ROLE_O_LABEL)
                cnt += 1
                # add argument roles
                if role in ont_dict[event_type]:
                    column_indexes.append(role2id[role_long])
                    cnt += 1
        row_indexes.extend([event_type_id] * cnt)
    i = torch.LongTensor([row_indexes, column_indexes])
    v = torch.LongTensor([1] * len(row_indexes))
    # print('i', i.tolist())
    # print('v', v.tolist())
    role_masks = (typestr2id, torch.sparse.FloatTensor(i, v, torch.Size([len(type2id), len(role2id)])).requires_grad_(False).to_dense().to(device))
    # role_masks = data_set.get_role_mask().to_dense()

    return role_masks


class EERunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test_ee", help="event extraction validation set")
        parser.add_argument("--train_ee", help="event extraction training set", required=False)
        parser.add_argument("--dev_ee", help="event extraction development set", required=False)
        parser.add_argument("--webd", help="word embedding", required=False)
        parser.add_argument("--ignore_time_test", help="testing ignore place in sr model", action='store_true')

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=1111, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=1, type=float)
        parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--finetune_classifier", help="pretrained model path")
        parser.add_argument("--earlystop", default=999999, type=int)
        parser.add_argument("--restart", default=999999, type=int)
        parser.add_argument("--shuffle", help="shuffle", action='store_true')
        parser.add_argument("--amr", help="use amr", action='store_true')

        parser.add_argument("--device", default="cpu")
        parser.add_argument("--hps", help="model hyperparams", required=False)

        self.a = parser.parse_args()

    def set_device(self, device="cpu"):
        # self.device = torch.device(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def get_tester(self, voc_i2s, voc_role_i2s):
        return EDTester(voc_i2s, voc_role_i2s, self.a.ignore_time_test)

    def run(self):
        print("Running on", self.a.device, ', optimizer=', self.a.optimizer, ', lr=%d' % self.a.lr)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False

        # create training set
        if self.a.train_ee:
            log('loading event extraction corpus from %s' % self.a.train_ee)

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
            LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL, vectors=pretrained_embedding)
            EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT, vectors=pretrained_embedding)
        else:
            WordsField.build_vocab(train_ee_set.WORDS, dev_ee_set.WORDS)
            LabelField.build_vocab(train_ee_set.LABEL, dev_ee_set.LABEL)
            EventsField.build_vocab(train_ee_set.EVENT, dev_ee_set.EVENT)
        PosTagsField.build_vocab(train_ee_set.POSTAGS, dev_ee_set.POSTAGS)
        EntityLabelsField.build_vocab(train_ee_set.ENTITYLABELS, dev_ee_set.ENTITYLABELS)


        consts.O_LABEL = LabelField.vocab.stoi[consts.O_LABEL_NAME]
        # print("O label is", consts.O_LABEL)
        consts.ROLE_O_LABEL = EventsField.vocab.stoi[consts.ROLE_O_LABEL_NAME]
        # print("O label for AE is", consts.ROLE_O_LABEL)

        dev_ee_set1 = ACE2005Dataset(path=self.a.dev_ee,
                                  fields={"words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          colcc: ("ADJM", AdjMatrixField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  amr=self.a.amr, keep_events=1, only_keep=True)

        test_ee_set1 = ACE2005Dataset(path=self.a.test_ee,
                                   fields={"words": ("WORDS", WordsField),
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

        self.a.label_weight = torch.ones([len(LabelField.vocab.itos)]) * 5
        self.a.label_weight[consts.O_LABEL] = 1.0
        self.a.arg_weight = torch.ones([len(EventsField.vocab.itos)]) * 5
        self.a.arg_weight[consts.ROLE_O_LABEL] = 1.0  #????
        # add role mask
        # self.a.role_mask = event_role_mask(self.a.test_ee, self.a.train_ee, self.a.dev_ee, LabelField.vocab.stoi, EventsField.vocab.stoi, self.device)
        self.a.role_mask = None

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
            log(model.parameters_requires_grads())
        else:
            model = load_ee_model(self.a.hps, None, WordsField.vocab.vectors, self.device)
            log('model created from scratch, there are %i sets of params' % len(model.parameters_requires_grads()))
            log(model.parameters_requires_grads())

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
        with open(os.path.join(self.a.out, "label_s2i.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "role_s2i.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "label_i2s.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.itos, f)
        with open(os.path.join(self.a.out, "role_i2s.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.itos, f)
        with open(os.path.join(self.a.out, "ee_hyps.json"), "w") as f:
            json.dump(self.a.hps, f)

        log('init complete\n')

        self.a.word_i2s = WordsField.vocab.itos
        self.a.label_i2s = LabelField.vocab.itos
        self.a.role_i2s = EventsField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        ee_train(
            model=model,
            train_set=train_ee_set,
            dev_set=dev_ee_set,
            test_set=test_ee_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a,
            other_testsets={
                "dev 1/1": dev_ee_set1,
                "test 1/1": test_ee_set1,
            },
            role_mask=self.a.role_mask
        )
        log('Done!')


if __name__ == "__main__":
    EERunner().run()
