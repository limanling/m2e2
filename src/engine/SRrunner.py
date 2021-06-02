import argparse
import os
import pickle
import json
import sys
from functools import partial

import numpy as np
import torch
from torch.nn import DataParallel
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchtext.data import Field
from torchtext.vocab import Vectors

import sys
sys.path.append('../..')

from src.dataflow.torch.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from src.util.vocab import Vocab
from src.models.sr import SRModel
from src.models.sr_object import SRModel_Object
from src.models.ace_classifier import ACEClassifier
from src.eval.SRtesting import SRTester
from src.engine.SRtraining import sr_train
from src.util.util_model import log
from src.dataflow.numpy.data_loader_situation import ImSituDataset
# from src.dataflow.numpy.data_loader_situation_objects import ImSituDataset

def load_sr_model(hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, fine_tune, device, ace_classifier=None, add_object=False, load_partial=False):
    if ace_classifier is None:
        ace_classifier = ACEClassifier(hps["wemb_dim"], hps["oc"], hps["ae_oc"], device)
    if add_object:
        mymodel = SRModel_Object(hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                              device, ace_classifier)
        if fine_tune is not None:
            mymodel.load_model(fine_tune, load_partial=load_partial)
            mymodel.to(device)
        return mymodel
    else:
        mymodel = SRModel(hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                          device, ace_classifier)
        if fine_tune is not None:
            mymodel.load_model(fine_tune, load_partial=load_partial)
            mymodel.to(device)
        return mymodel

class SRRunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test_sr", help="situation recognition validation set")
        parser.add_argument("--train_sr", help="situation recognition training set", required=False)
        parser.add_argument("--dev_sr", help="situation recognition development set", required=False)
        parser.add_argument("--train_ee", help="ace training set to load the vocab", required=False)
        parser.add_argument("--webd", help="word embedding", required=False)
        parser.add_argument("--wnebd", help="situation recognition noun word embedding", required=False)
        parser.add_argument("--wvebd", help="situation recognition verb word embedding", required=False)
        parser.add_argument("--wrebd", help="situation recognition role word embedding", required=False)
        parser.add_argument("--vocab", help="situation recognition vocab_dir", required=False)
        parser.add_argument("--image_dir", help="situation recognition image_dir", required=False)
        parser.add_argument("--imsitu_ontology_file", help="imsitu_ontology_file", required=False)
        parser.add_argument("--object_class_map_file", help="object_class_map_file", required=False)
        parser.add_argument("--object_detection_pkl_file", help="object_detection_pkl_file", required=False)
        parser.add_argument("--object_detection_threshold", default=0.2, type=float, help="object_detection_threshold", required=False)
        parser.add_argument("--verb_mapping_file", help="verb_mapping_file", required=False)
        parser.add_argument("--add_object", help="add_object", action='store_true')
        parser.add_argument("--train_ace", help="train_ace", action='store_true')
        parser.add_argument("--visualize_path", help="visualize_path", required=False)

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--earlystop", default=999999, type=int)
        parser.add_argument("--restart", default=999999, type=int)
        parser.add_argument("--shuffle", help="shuffle", action='store_true')
        parser.add_argument("--textontology", help="use which ontology to train SR", action='store_true')

        parser.add_argument("--device", default="cpu")
        parser.add_argument("--hps", help="model hyperparams", required=False)

        parser.add_argument("--filter_irrelevant_verbs", help="filter_irrelevant_verbs", action='store_true')
        parser.add_argument("--filter_place", help="filter_place", action='store_true')

        self.a = parser.parse_args()
        print('self.a.vocab', self.a.vocab)
        self.vocab_noun = Vocab(os.path.join(self.a.vocab, 'vocab_situation_noun.pkl'), load=True)
        self.vocab_role = Vocab(os.path.join(self.a.vocab, 'vocab_situation_role.pkl'), load=True)
        self.vocab_verb = Vocab(os.path.join(self.a.vocab, 'vocab_situation_verb.pkl'), load=True)
        # print('self.a.hps', self.a)
        # print('self.a[\'hps]\'', self.a['hps'])
        print('self.a.hps', self.a.hps)
        # self.a.hps['wvemb_size'] = self.vocab_verb.size
        # self.a.hps['wremb_size'] = self.vocab_role.size
        # self.a.hps['wnemb_size'] = self.vocab_noun.size

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])


    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device



    def get_tester(self):
        return SRTester()

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True

        # build text event vocab and ee_role vocab
        WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        PosTagsField = Field(lower=True, batch_first=True)
        EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
        AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)
        # only for ee
        LabelField = Field(lower=False, batch_first=True, pad_token='0', unk_token=None)
        EventsField = EventField(lower=False, batch_first=True)
        SENTIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        colcc = 'combined-parsing'
        train_ee_set = ACE2005Dataset(path=self.a.train_ee,
                                      fields={"sentence_id": ("SENTID", SENTIDField),
                                              "words": ("WORDS", WordsField),
                                              "pos-tags": ("POSTAGS", PosTagsField),
                                              "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                              colcc: ("ADJM", AdjMatrixField),
                                              "golden-event-mentions": ("LABEL", LabelField),
                                              "all-events": ("EVENT", EventsField),
                                              "all-entities": ("ENTITIES", EntitiesField)
                                              },
                                      amr=False, keep_events=1)
        pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
        LabelField.build_vocab(train_ee_set.LABEL, vectors=pretrained_embedding)
        EventsField.build_vocab(train_ee_set.EVENT, vectors=pretrained_embedding)

        # consts.O_LABEL = LabelField.vocab.stoi["O"]
        # # print("O label is", consts.O_LABEL)
        # consts.ROLE_O_LABEL = EventsField.vocab.stoi["OTHER"]
        # print("O label for AE is", consts.ROLE_O_LABEL)

        # create training set
        if self.a.train_sr:
            log('loading corpus from %s' % self.a.train_sr)

        train_sr_set = ImSituDataset(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, 
                                     LabelField.vocab.stoi, EventsField.vocab.stoi,
                                 self.a.imsitu_ontology_file,
                                 self.a.train_sr, self.a.verb_mapping_file,
                                 self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                 self.a.object_detection_threshold,
                                 self.transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                     load_object=self.a.add_object, filter_place=self.a.filter_place)
        dev_sr_set = ImSituDataset(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, 
                                   LabelField.vocab.stoi, EventsField.vocab.stoi,
                                 self.a.imsitu_ontology_file,
                                 self.a.dev_sr, self.a.verb_mapping_file,
                                 self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                 self.a.object_detection_threshold,
                                 self.transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                   load_object=self.a.add_object, filter_place=self.a.filter_place)
        test_sr_set = ImSituDataset(self.a.image_dir, self.vocab_noun, self.vocab_role, self.vocab_verb, 
                                    LabelField.vocab.stoi, EventsField.vocab.stoi,
                                 self.a.imsitu_ontology_file,
                                 self.a.test_sr, self.a.verb_mapping_file,
                                 self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                 self.a.object_detection_threshold,
                                 self.transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                    load_object=self.a.add_object, filter_place=self.a.filter_place)


        embeddingMatrix_noun = torch.FloatTensor(np.load(self.a.wnebd)).to(self.device)
        embeddingMatrix_verb = torch.FloatTensor(np.load(self.a.wvebd)).to(self.device)
        embeddingMatrix_role = torch.FloatTensor(np.load(self.a.wrebd)).to(self.device)
        # consts.O_LABEL = self.vocab_verb.word2id['0'] # verb??
        # consts.ROLE_O_LABEL = self.vocab_role.word2id["OTHER"] #???

        # self.a.label_weight = torch.ones([len(vocab_sr.id2word)]) * 5 # more important to learn
        # self.a.label_weight[consts.O_LABEL] = 1.0 #???

        self.a.hps = eval(self.a.hps)
        if self.a.textontology:
            if "wvemb_size" not in self.a.hps:
                self.a.hps["wvemb_size"] = len(LabelField.vocab.stoi)
            if "wremb_size" not in self.a.hps:
                self.a.hps["wremb_size"] = len(EventsField.vocab.itos)
            if "wnemb_size" not in self.a.hps:
                self.a.hps["wnemb_size"] = len(self.vocab_noun.id2word)
            if "oc" not in self.a.hps:
                self.a.hps["oc"] = len(LabelField.vocab.itos)
            if "ae_oc" not in self.a.hps:
                self.a.hps["ae_oc"] = len(EventsField.vocab.itos)
        else:
            if "wvemb_size" not in self.a.hps:
                self.a.hps["wvemb_size"] = len(self.vocab_verb.id2word)
            if "wremb_size" not in self.a.hps:
                self.a.hps["wremb_size"] = len(self.vocab_role.id2word)
            if "wnemb_size" not in self.a.hps:
                self.a.hps["wnemb_size"] = len(self.vocab_noun.id2word)
            if "oc" not in self.a.hps:
                self.a.hps["oc"] = len(LabelField.vocab.itos)
            if "ae_oc" not in self.a.hps:
                self.a.hps["ae_oc"] = len(EventsField.vocab.itos)

        tester = self.get_tester()

        if self.a.textontology:
            if self.a.finetune:
                log('init model from ' + self.a.finetune)
                model = load_sr_model(self.a.hps, embeddingMatrix_noun, LabelField.vocab.vectors, EventsField.vocab.vectors,
                                      self.a.finetune, self.device, add_object=self.a.add_object)
                log('sr model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
            else:
                model = load_sr_model(self.a.hps, embeddingMatrix_noun, LabelField.vocab.vectors, EventsField.vocab.vectors, None, self.device, add_object=self.a.add_object)
                log('sr model created from scratch, there are %i sets of params' % len(model.parameters_requires_grads()))
        else:
            if self.a.finetune:
                log('init model from ' + self.a.finetune)
                model = load_sr_model(self.a.hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                                      self.a.finetune, self.device, add_object=self.a.add_object)
                log('sr model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
            else:
                model = load_sr_model(self.a.hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                                      None, self.device, add_object=self.a.add_object)
                log('sr model created from scratch, there are %i sets of params' % len(
                    model.parameters_requires_grads()))

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

        # for name, para in model.named_parameters():
        #     if para.requires_grad:
        #         print(name)
        # exit(1)

        log('optimizer in use: %s' % str(self.a.optimizer))

        log('init complete\n')

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)

        self.a.word_i2s = self.vocab_noun.id2word
        # if self.a.textontology:
        self.a.acelabel_i2s = LabelField.vocab.itos
        self.a.acerole_i2s = EventsField.vocab.itos
        with open(os.path.join(self.a.out, "label_s2i.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "role_s2i.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "label_i2s.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.itos, f)
        with open(os.path.join(self.a.out, "role_i2s.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.itos, f)
        # else:
        self.a.label_i2s = self.vocab_verb.id2word #LabelField.vocab.itos
        self.a.role_i2s = self.vocab_role.id2word
            # save as Vocab
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer


        with open(os.path.join(self.a.out, "sr_hyps.json"), "w") as f:
            json.dump(self.a.hps, f)

        sr_train(
            model=model,
            train_set=train_sr_set,
            dev_set=dev_sr_set,
            test_set=test_sr_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a,
            other_testsets={
                # "dev 1/1":  dev_sr_loader,
                # "test 1/1": test_sr_loader,
            }
        )
        log('Done!')


if __name__ == "__main__":
    SRRunner().run()