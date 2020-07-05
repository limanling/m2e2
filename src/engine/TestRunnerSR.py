import argparse
import os
import pickle
import json
import sys
from functools import partial

import numpy as np
from math import ceil
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
from src.models.ace_classifier import ACEClassifier
from src.eval.SRtesting import SRTester
from src.engine.SRtraining import run_over_data_sr
from src.util.util_model import log
from src.dataflow.numpy.data_loader_situation import ImSituDataset
from src.engine.SRrunner import load_sr_model
from src.dataflow.numpy.data_loader_situation import unpack, image_collate_fn

class TestSRRunner(object):
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
        parser.add_argument("--hps_path", help="model hyperparams", required=False)

        parser.add_argument("--filter_irrelevant_verbs", help="filter_irrelevant_verbs", action='store_true')
        parser.add_argument("--filter_place", help="filter_place", action='store_true')

        self.a = parser.parse_args()
        if self.a.hps_path:
            self.a.hps = json.load(open(self.a.hps_path))
        print(self.a.hps)



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
        colcc = 'stanford-colcc'
        train_ee_set = ACE2005Dataset(path=self.a.train_ee,
                                      fields={
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

        # consts.O_LABEL = LabelField.vocab.stoi[consts.O_LABEL_NAME]
        # # print("O label is", consts.O_LABEL)
        # consts.ROLE_O_LABEL = EventsField.vocab.stoi[consts.ROLE_O_LABEL_NAME]
        # # print("O label for AE is", consts.ROLE_O_LABEL)

        # create testing set
        if self.a.test_sr:
            log('loading corpus from %s' % self.a.test_sr)

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

        # train_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
        #                              EventsField.vocab.stoi, LabelField.vocab.stoi,
        #                          self.a.imsitu_ontology_file,
        #                          self.a.train_sr, self.a.verb_mapping_file,
        #                          self.a.object_class_map_file, self.a.object_detection_pkl_file,
        #                          self.a.object_detection_threshold,
        #                          transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
        #                              load_object=self.a.add_object, filter_place=self.a.filter_place)
        # dev_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
        #                            EventsField.vocab.stoi, LabelField.vocab.stoi,
        #                          self.a.imsitu_ontology_file,
        #                          self.a.dev_sr, self.a.verb_mapping_file,
        #                          self.a.object_class_map_file, self.a.object_detection_pkl_file,
        #                          self.a.object_detection_threshold,
        #                          transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
        #                            load_object=self.a.add_object, filter_place=self.a.filter_place)
        test_sr_set = ImSituDataset(self.a.image_dir, vocab_noun, vocab_role, vocab_verb,
                                    EventsField.vocab.stoi, LabelField.vocab.stoi,
                                 self.a.imsitu_ontology_file,
                                 self.a.test_sr, self.a.verb_mapping_file,
                                 self.a.object_class_map_file, self.a.object_detection_pkl_file,
                                 self.a.object_detection_threshold,
                                 transform, filter_irrelevant_verbs=self.a.filter_irrelevant_verbs,
                                    load_object=self.a.add_object, filter_place=self.a.filter_place)


        embeddingMatrix_noun = torch.FloatTensor(np.load(self.a.wnebd)).to(self.device)
        embeddingMatrix_verb = torch.FloatTensor(np.load(self.a.wvebd)).to(self.device)
        embeddingMatrix_role = torch.FloatTensor(np.load(self.a.wrebd)).to(self.device)
        # consts.O_LABEL = vocab_verb.word2id['0'] # verb??
        # consts.ROLE_O_LABEL = vocab_role.word2id["OTHER"] #???

        # self.a.label_weight = torch.ones([len(vocab_sr.id2word)]) * 5 # more important to learn
        # self.a.label_weight[consts.O_LABEL] = 1.0 #???

        if not self.a.hps_path:
            self.a.hps = eval(self.a.hps)
        if self.a.textontology:
            if "wvemb_size" not in self.a.hps:
                self.a.hps["wvemb_size"] = len(LabelField.vocab.stoi)
            if "wremb_size" not in self.a.hps:
                self.a.hps["wremb_size"] = len(EventsField.vocab.itos)
            if "wnemb_size" not in self.a.hps:
                self.a.hps["wnemb_size"] = len(vocab_noun.id2word)
            if "oc" not in self.a.hps:
                self.a.hps["oc"] = len(LabelField.vocab.itos)
            if "ae_oc" not in self.a.hps:
                self.a.hps["ae_oc"] = len(EventsField.vocab.itos)
        else:
            if "wvemb_size" not in self.a.hps:
                self.a.hps["wvemb_size"] = len(vocab_verb.id2word)
            if "wremb_size" not in self.a.hps:
                self.a.hps["wremb_size"] = len(vocab_role.id2word)
            if "wnemb_size" not in self.a.hps:
                self.a.hps["wnemb_size"] = len(vocab_noun.id2word)
            if "oc" not in self.a.hps:
                self.a.hps["oc"] = len(LabelField.vocab.itos)
            if "ae_oc" not in self.a.hps:
                self.a.hps["ae_oc"] = len(EventsField.vocab.itos)

        tester = self.get_tester()

        if self.a.textontology:
            if self.a.finetune:
                log('init model from ' + self.a.finetune)
                model = load_sr_model(self.a.hps, embeddingMatrix_noun, LabelField.vocab.vectors, EventsField.vocab.vectors, self.a.finetune, self.device, add_object=self.a.add_object)
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

        # for name, para in model.named_parameters():
        #     if para.requires_grad:
        #         print(name)
        # exit(1)

        log('init complete\n')

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)

        self.a.word_i2s = vocab_noun.id2word
        # if self.a.textontology:
        self.a.acelabel_i2s = LabelField.vocab.itos
        self.a.acerole_i2s = EventsField.vocab.itos
        # with open(os.path.join(self.a.out, "label_s2i.vec"), "wb") as f:
        #     pickle.dump(LabelField.vocab.stoi, f)
        # with open(os.path.join(self.a.out, "role_s2i.vec"), "wb") as f:
        #     pickle.dump(EventsField.vocab.stoi, f)
        # with open(os.path.join(self.a.out, "label_i2s.vec"), "wb") as f:
        #     pickle.dump(LabelField.vocab.itos, f)
        # with open(os.path.join(self.a.out, "role_i2s.vec"), "wb") as f:
        #     pickle.dump(EventsField.vocab.itos, f)
        # else:
        self.a.label_i2s = vocab_verb.id2word #LabelField.vocab.itos
        self.a.role_i2s = vocab_role.id2word
            # save as Vocab
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer


        # with open(os.path.join(self.a.out, "sr_hyps.json"), "w") as f:
        #     json.dump(self.a.hps, f)

        test_iter = torch.utils.data.DataLoader(dataset=test_sr_set,
                                                batch_size=self.a.batch,
                                                shuffle=False,
                                                num_workers=2,
                                                collate_fn=image_collate_fn)

        verb_roles = test_sr_set.get_verb_role_mapping()

        if 'visualize_path' not in self.a:
            visualize_path = None
        else:
            visualize_path = self.a.visualize_path

        test_loss, test_verb_p, test_verb_r, test_verb_f1, \
        test_role_p, test_role_r, test_role_f1, \
        test_noun_p, test_noun_r, test_noun_f1, \
        test_triple_p, test_triple_r, test_triple_f1, \
        test_noun_p_relaxed, test_noun_r_relaxed, test_noun_f1_relaxed, \
        test_triple_p_relaxed, test_triple_r_relaxed, test_triple_f1_relaxed = run_over_data_sr(data_iter=test_iter,
                                                                                                optimizer=None,
                                                                                                model=model,
                                                                                                need_backward=False,
                                                                                                MAX_STEP=ceil(len(
                                                                                                    test_sr_set) / self.a.batch),
                                                                                                tester=tester,
                                                                                                hyps=model.hyperparams,
                                                                                                device=model.device,
                                                                                                maxnorm=self.a.maxnorm,
                                                                                                word_i2s=self.a.word_i2s,
                                                                                                label_i2s=self.a.label_i2s,
                                                                                                role_i2s=self.a.role_i2s,
                                                                                                verb_roles=verb_roles,
                                                                                                load_object=self.a.add_object,
                                                                                                visualize_path=visualize_path,
                                                                                                save_output=os.path.join(
                                                                                                    self.a.out,
                                                                                                    "test_final.txt"))
        print("\nFinally test loss: ", test_loss,
              "\ntest verb p: ", test_verb_p,
              " test verb r: ", test_verb_r,
              " test verb f1: ", test_verb_f1,
              "\ntest role p: ", test_role_p,
              " test role r: ", test_role_r,
              " test role f1: ", test_role_f1,
              "\ntest noun p: ", test_noun_p,
              " test noun r: ", test_noun_r,
              " test noun f1: ", test_noun_f1,
              "\ntest triple p: ", test_triple_p,
              " test triple r: ", test_triple_r,
              " test triple f1: ", test_triple_f1,
              "\ntest noun p relaxed: ", test_noun_p_relaxed,
              " test noun r relaxed: ", test_noun_r_relaxed,
              " test noun f1 relaxed: ", test_noun_f1_relaxed,
              "\ntest triple p relaxed: ", test_triple_p_relaxed,
              " test triple r relaxed: ", test_triple_r_relaxed,
              " test triple f1 relaxed: ", test_triple_f1_relaxed
              )


if __name__ == "__main__":
    TestSRRunner().run()