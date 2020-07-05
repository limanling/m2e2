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
from src.dataflow.torch.Data import MultiTokenField, SparseField, EntityField
from torchtext.data import Field
from src.util.vocab import Vocab
from torchtext.vocab import Vectors
from src.dataflow.numpy.data_loader_grounding import GroundingDataset
from src.models.grounding import GroundingModel
from src.eval.Groundingtesting import GroundingTester
from src.engine.Groundingtraining import grounding_train
from src.engine.SRrunner import load_sr_model
from src.engine.EErunner import load_ee_model
from src.util.util_model import log


class GroundingRunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test", help="validation set")
        parser.add_argument("--train", help="training set", required=False)
        parser.add_argument("--dev", help="development set", required=False)
        parser.add_argument("--webd", help="word embedding", required=False)
        parser.add_argument("--img_dir", help="Grounding images directory", required=False)
        parser.add_argument("--amr", help="use amr", action='store_true')

        # sr model parameter
        parser.add_argument("--wnebd", help="noun word embedding", required=False)
        parser.add_argument("--wvebd", help="verb word embedding", required=False)
        parser.add_argument("--wrebd", help="role word embedding", required=False)
        parser.add_argument("--add_object", help="add_object", action='store_true')
        parser.add_argument("--object_class_map_file", help="object_class_map_file", required=False)
        parser.add_argument("--object_detection_pkl_file", help="object_detection_pkl_file", required=False)
        parser.add_argument("--object_detection_threshold", default=0.2, type=float, help="object_detection_threshold",
                            required=False)

        parser.add_argument("--vocab", help="vocab_dir", required=False)
        parser.add_argument("--sr_hps", help="sr model hyperparams", required=False)

        # ee model parameter
        parser.add_argument("--ee_hps", help="ee model hyperparams", required=False)

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune_sr", help="pretrained sr model path")
        parser.add_argument("--finetune_ee", help="pretrained ee model path")
        parser.add_argument("--earlystop", default=999999, type=int)
        parser.add_argument("--restart", default=999999, type=int)
        parser.add_argument("--shuffle", help="shuffle", action='store_true')

        parser.add_argument("--device", default="cpu")

        self.a = parser.parse_args()

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

    def get_tester(self):
        return GroundingTester()

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)
        torch.backends.cudnn.benchmark = True

        # create training set
        if self.a.train:
            log('loading corpus from %s' % self.a.train)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        IMAGEIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        SENTIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        # IMAGEField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        PosTagsField = Field(lower=True, batch_first=True)
        EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
        AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)

        if self.a.amr:
            colcc = 'amr-colcc'
        else:
            colcc = 'stanford-colcc'
        print(colcc)

        train_set = GroundingDataset(path=self.a.train,
                                     img_dir=self.a.img_dir,
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
                                     object_detection_pkl_file=self.a.object_detection_pkl_file,
                                     object_detection_threshold=self.a.object_detection_threshold,
                                     )

        dev_set = GroundingDataset(path=self.a.dev,
                                   img_dir=self.a.img_dir,
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
                                   object_detection_pkl_file=self.a.object_detection_pkl_file,
                                   object_detection_threshold=self.a.object_detection_threshold,
                                   )

        test_set = GroundingDataset(path=self.a.test,
                                    img_dir=self.a.img_dir,
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
                                    object_detection_pkl_file=self.a.object_detection_pkl_file,
                                    object_detection_threshold=self.a.object_detection_threshold,
                                    )

        if self.a.webd:
            pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
            WordsField.build_vocab(train_set.WORDS, dev_set.WORDS, vectors=pretrained_embedding)
        else:
            WordsField.build_vocab(train_set.WORDS, dev_set.WORDS)
        # WordsField.build_vocab(train_set.WORDS, dev_set.WORDS)
        PosTagsField.build_vocab(train_set.POSTAGS, dev_set.POSTAGS)
        EntityLabelsField.build_vocab(train_set.ENTITYLABELS, dev_set.ENTITYLABELS)

        # sr model initialization
        self.a.sr_hps = eval(self.a.sr_hps)
        vocab_noun = Vocab(os.path.join(self.a.vocab, 'vocab_situation_noun.pkl'), load=True)
        vocab_role = Vocab(os.path.join(self.a.vocab, 'vocab_situation_role.pkl'), load=True)
        vocab_verb = Vocab(os.path.join(self.a.vocab, 'vocab_situation_verb.pkl'), load=True)
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

        self.a.ee_hps = eval(self.a.ee_hps)
        if "wemb_size" not in self.a.ee_hps:
            self.a.ee_hps["wemb_size"] = len(WordsField.vocab.itos)
        if "pemb_size" not in self.a.ee_hps:
            self.a.ee_hps["pemb_size"] = len(PosTagsField.vocab.itos)
        if "psemb_size" not in self.a.ee_hps:
            self.a.ee_hps["psemb_size"] = max([train_set.longest(), dev_set.longest(), test_set.longest()]) + 2
        if "eemb_size" not in self.a.ee_hps:
            self.a.ee_hps["eemb_size"] = len(EntityLabelsField.vocab.itos)
        if "oc" not in self.a.ee_hps:
            self.a.ee_hps["oc"] = 36 #???
        if "ae_oc" not in self.a.ee_hps:
            self.a.ee_hps["ae_oc"] = 20 #???

        tester = self.get_tester()

        if self.a.finetune_sr:
            log('init sr model from ' + self.a.finetune_sr)
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, self.a.finetune_sr, self.device)
            log('sr model loaded, there are %i sets of params' % len(sr_model.parameters_requires_grads()))
        else:
            sr_model = load_sr_model(self.a.sr_hps, embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role, None, self.device)
            log('sr model created from scratch, there are %i sets of params' % len(sr_model.parameters_requires_grads()))

        if self.a.finetune_ee:
            log('init model from ' + self.a.finetune_ee)
            ee_model = load_ee_model(self.a.ee_hps, self.a.finetune_ee, WordsField.vocab.vectors, self.device)
            log('model loaded, there are %i sets of params' % len(ee_model.parameters_requires_grads()))
        else:
            ee_model = load_ee_model(self.a.ee_hps, None, WordsField.vocab.vectors, self.device)
            log('model created from scratch, there are %i sets of params' % len(ee_model.parameters_requires_grads()))

        model = GroundingModel(ee_model, sr_model, self.get_device())

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
        with open(os.path.join(self.a.out, "ee_hyps.json"), "w") as f:
            json.dump(self.a.ee_hps, f)
        with open(os.path.join(self.a.out, "sr_hyps.json"), "w") as f:
            json.dump(self.a.sr_hps, f)

        log('init complete\n')

        self.a.word_i2s = vocab_noun.id2word
        self.a.label_i2s = vocab_verb.id2word  # LabelField.vocab.itos
        self.a.role_i2s = vocab_role.id2word
        self.a.word_i2s = WordsField.vocab.itos
        # self.a.label_i2s = LabelField.vocab.itos
        # self.a.role_i2s = EventsField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        grounding_train(
            model=model,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a,
            other_testsets={
                # "dev 1/1": dev_set1,
                # "test 1/1": test_set1,
            },
            transform=transform,
            vocab_objlabel=vocab_noun.word2id
        )
        log('Done!')


if __name__ == "__main__":
    GroundingRunner().run()
