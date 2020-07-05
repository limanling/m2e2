import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import numpy as np
import copy

from torchnlp.nn.attention import Attention

from src.util import consts
from src.models.modules.DynamicLSTM import DynamicLSTM
from src.models.modules.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer
from src.models.modules.EmbeddingLayerImage import EmbeddingLayerImage
from src.models.modules.GCN import GraphConvolution
from src.models.modules.HighWay import HighWay
from src.models.modules.model import Model
from src.models.ee import EDModel
from src.models.sr import SRModel
from src.eval.EEtesting import EDTester

class GroundingModel(Model):
    def __init__(self, ed_model, sr_model, device=torch.device("cpu")):
        super(GroundingModel, self).__init__()

        self.device = device

        self.ed_model = ed_model
        self.sr_model = sr_model
        # self.word2verb_att = Attention(sr_model.get_common_dim(), attention_type='general')
        # self.word2noun_att = Attention(sr_model.get_common_dim(), attention_type='general')

        # # Event Extractor
        # self.ol, self.ae_ol = ed_model.get_extractor()  # BottledXavierLinear(in_features=4 * hyps["lstm_dim"], out_features=hyps["ae_oc"]).to(device=device)

        self.to(device)

    def get_sr_model(self):
        return self.sr_model

    def get_ee_model(self):
        return self.ed_model

    def forward(self, word_sequence, x_len,
                pos_tagging_sequence, entity_type_sequence, adj,
                img_id_batch, image_batch,
                add_object=False, bbox_entities_id=None, bbox_entities_region=None,
                bbox_entities_label=None, object_num_batch=None
                ):

        BATCH_SIZE = image_batch.size(0)
        if not add_object:
            OBJECT_LEN = self.sr_model.get_role_num()
            SEQ_LEN = OBJECT_LEN + 1
        else:
            OBJECT_LEN = bbox_entities_region.size()[-4]
            SEQ_LEN = OBJECT_LEN + 1 # the num of the GCN nodes
        # bbox_entities_id = None
        # bbox_entities_region = None
        # bbox_entities_label = None
        # object_num_batch = None

        verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = self.sr_model.get_common_feature(img_id_batch, image_batch,
                           bbox_entities_id, bbox_entities_region, bbox_entities_label,
                           object_num_batch, BATCH_SIZE, OBJECT_LEN, SEQ_LEN)

        # print('verb_emb', verb_emb.size())  #[batch, 300]
        # print('noun_emb', noun_emb.size())  #[batch, 192, 300]

        word_common, word_mask, word_emb = self.ed_model.get_common_feature(word_sequence, x_len, pos_tagging_sequence, entity_type_sequence, adj)

        image_common, sent_common, word2noun_att_output, word_common, noun2word_att_output, noun_emb_common = self.similarity(verb_emb_common, noun_emb_common, verb_emb, noun_emb, word_common, word_emb, word_mask)
        return image_common, sent_common, word2noun_att_output, word_common, noun2word_att_output, noun_emb_common

    def similarity(self, verb_emb_common, noun_emb_common, verb_emb, noun_emb, word_common, word_emb, word_mask):
        '''

        :param verb_emb_common: batch, d
        :param noun_emb_common: batch, 192, d
        :param verb_emb: batch, d
        :param noun_emb: batch, 192, d
        :param word_common: batch, seq_len, d'
        :param word_emb: batch, seq_len, d'
        :param word_mask:
        :return:
        '''

        image_common_verb = verb_emb_common

        # word4att = word_common  # word_emb
        # noun4att = noun_emb_common
        # word_common: (batch_size, seq_len, d')
        # word_emb: (batch_size, seq_len, d)

        # (1) noun - entity alignment
        # --> noun_emb (batch_size, 192, emb) - word_common (batch_size, seq_len, d')
        # triplet loss? can only find negative word, out of sentence, but it may not the real negative
        # print('verb_emb_common', verb_emb_common.size())
        # print('noun_emb_common', noun_emb_common.size())
        # print('word_common', word_common.size())  # [1, 2, 300]
        # print('word_common.size[-1]', word_common.size()[-1])
        # print('noun_emb_common', noun_emb_common.size())  # [4, 192, 300]
        noun2word_att_weights_ = torch.matmul(noun_emb_common, torch.transpose(word_common, len(word_common.size())-2,
                                                                               len(word_common.size())-1 ))   # batch_size, 192, seq_len
        noun2word_att_weights = F.softmax(noun2word_att_weights_, dim=-1)  # batch_size, 192, seq_len
        noun2word_att_output = torch.matmul(noun2word_att_weights, word_common)  # batch_size, 192, d'
        image_common_noun = torch.mean(noun2word_att_output, dim=1)  # (batch_size, d')

        word2noun_att_weights_ = torch.transpose(noun2word_att_weights_, 1, 2)  # batch_size, seq_len, 192
        # word2noun_att_weights_ = torch.matmul(word_common, torch.transpose(noun_emb_common, 1,
        #                                                              2))  # batch_size, seq_len, 192 (batch_size, seq_len, d') * (batch_size, d, 192)
        word2noun_att_weights = F.softmax(word2noun_att_weights_, dim=-1)  # batch_size, seq_len, 192
        word2noun_att_output = torch.matmul(word2noun_att_weights, noun_emb_common)  # batch_size, seq_len, d
        # word2noun_att_output, word2noun_att_weights = self.word2noun_att(word_common, noun_emb)  #(batch_size, seq_len, d), (batch_size, seq_len, 192)
        sent_common_noun = torch.mean(word2noun_att_output, dim=1)  # (batch_size, d)


        # (2) (verb, role, noun) - (dep, edge, gov) alignment

        # (3) caption - image alignment
        verb2word_att_weights_ = torch.matmul(verb_emb_common.unsqueeze(1), torch.transpose(word_common, len(word_common.size())-2,
                                                                                            len(word_common.size())-1 ))  # (batch_size, 1, d), (batch_size, d', seq_len)
        verb2word_att_weights = F.softmax(verb2word_att_weights_, dim=-1)  # (batch_size, 1, seq_len)
        sent_common_verb = torch.matmul(verb2word_att_weights, word_common).squeeze(1)  # (batch_size, 1, d')
        # sent_common_verb = torch.mean(word2verb_att_output, dim=1)  # (batch_size, d')
        # word2verb_att_output, word2verb_att_weights = self.word2verb_att(word_common, verb_emb.unsqueeze(1))  # (batch_size, seq_len, d'), (batch_size, seq_len, 1)
        # sent_common_verb = torch.mean(word2verb_att_output, dim=1)  # (batch_size, d')

        # image_common = torch.cat([image_common_verb, image_common_noun], dim=-1)
        # sent_common = torch.cat([sent_common_verb, sent_common_noun], dim=-1)
        image_common = image_common_verb
        sent_common = sent_common_verb

        return image_common, sent_common, word2noun_att_output, word_common, noun2word_att_output, noun_emb_common

    def calculate_loss_grounding(self, emb_sentence, emb_image, word2noun_att_output, word_common, noun2word_att_output, noun_emb_common):
        # return F.mse_loss(emb_sentence, emb_image)
        BATCH_SIZE = emb_sentence.size(0)
        caption_logits = torch.mm(emb_image, emb_sentence.t())  # [batch,emb] * [emb,sent_num]
        caption_logits = F.log_softmax(caption_logits, dim=-1)
        idx_caption_gt = torch.arange(BATCH_SIZE)
        cost_caption = torch.mean(- caption_logits[torch.arange(BATCH_SIZE), idx_caption_gt])

        image_logits = F.log_softmax(caption_logits.t(), dim=-1)
        idx_iamge_gt = torch.arange(BATCH_SIZE)
        cost_image = torch.mean(- image_logits[torch.arange(BATCH_SIZE), idx_iamge_gt])

        cost_noun = F.mse_loss(word2noun_att_output, word_common)
        cost_word = F.mse_loss(noun2word_att_output, noun_emb_common)

        loss_terms = {'caption': cost_caption.item(), 'image': cost_image.item()}
        loss_grouding = cost_caption + cost_image + cost_noun + cost_word
        return loss_grouding, loss_terms, caption_logits, image_logits

    def calculate_loss_grounding_single(self, emb_sentence, emb_image):
        # return F.mse_loss(emb_sentence, emb_image)
        BATCH_SIZE = emb_sentence.size(0)
        caption_logits = torch.mm(emb_image, emb_sentence.t())
        caption_logits = F.log_softmax(caption_logits)
        idx_caption_gt = torch.arange(BATCH_SIZE)
        cost_caption = torch.mean(- caption_logits[torch.arange(BATCH_SIZE), idx_caption_gt])

        # image_logits = F.log_softmax(caption_logits.t())
        # idx_iamge_gt = torch.arange(BATCH_SIZE)
        # cost_image = torch.mean(- image_logits[torch.arange(BATCH_SIZE), idx_iamge_gt])

        loss_terms = {'caption': cost_caption.item()}
        loss_grouding = cost_caption
        return loss_grouding, loss_terms, caption_logits

    def predict_image(self, doc_image_result, batch_id, img_id_batch, image_batch,
                      bbox_entities_id, bbox_entities_region, bbox_entities_label,
                      object_num_batch, OBJECT_LEN, SEQ_LEN,
                      all_verb_embs, all_noun_embs, role_masks,
                      y_verb_all_, y_role_all_, heatmap_all, image_id_all):
        '''
        multiple images in one data instance
        :param batch_id:
        :param img_id_batch:
        :param image_batch:
        :param bbox_entities_id:
        :param bbox_entities_region:
        :param bbox_entities_label:
        :param object_num_batch:
        :param OBJECT_LEN:
        :param SEQ_LEN:
        :param all_verb_embs:
        :param all_noun_embs:
        :param role_masks:
        :param y_verb_all_:
        :param y_role_all_:
        :param heatmap_all:
        :param image_id_all:
        :return:
        '''

        # whether the document has been processed
        image_id_first = img_id_batch[batch_id][0]
        docid = image_id_first[:image_id_first.rfind('_')]
        if docid in doc_image_result:
            verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, image_id_all = doc_image_result[docid]
            return verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, image_id_all

        IMG_NUM = len(img_id_batch[batch_id])
        if bbox_entities_id is None:
            verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = self.sr_model.get_common_feature(
                img_id_batch[batch_id][:IMG_NUM], image_batch[batch_id][:IMG_NUM],
                None, None, None, None,
                IMG_NUM, OBJECT_LEN, SEQ_LEN)
        else:
            verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = self.sr_model.get_common_feature(
                img_id_batch[batch_id][:IMG_NUM], image_batch[batch_id][:IMG_NUM],
                bbox_entities_id[batch_id][:IMG_NUM], bbox_entities_region[batch_id][:IMG_NUM],
                bbox_entities_label[batch_id][:IMG_NUM], object_num_batch[batch_id][:IMG_NUM],
                IMG_NUM, OBJECT_LEN, SEQ_LEN)

        # print('heatmap_org', heatmap)

        verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, \
            image_id_all = self.image_inference(verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap,
                    doc_image_result, batch_id, img_id_batch, image_batch,
                    bbox_entities_id, bbox_entities_region, bbox_entities_label,
                    object_num_batch, OBJECT_LEN, SEQ_LEN,
                    all_verb_embs, all_noun_embs, role_masks,
                    y_verb_all_, y_role_all_, heatmap_all, image_id_all,
                    IMG_NUM, docid
        )

        return verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, image_id_all

    def image_inference(self, verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap,
                        doc_image_result, batch_id, img_id_batch, image_batch,
                        bbox_entities_id, bbox_entities_region, bbox_entities_label,
                        object_num_batch, OBJECT_LEN, SEQ_LEN,
                        all_verb_embs, all_noun_embs, role_masks,
                        y_verb_all_, y_role_all_, heatmap_all, image_id_all,
                        IMG_NUM, docid):

        # get image prediction
        # image prediction (multiple images)
        # use attention, do not need padding for nouns
        verb_logits = torch.mm(verb_emb[:IMG_NUM], all_verb_embs.t())
        verb_logits = F.log_softmax(verb_logits, dim=-1)
        y_verb_ = torch.max(verb_logits, dim=-1)[1]
        y_verb_all_.append(y_verb_)  # not extend, list of list

        # OBJ_NUM = noun_emb.size(1)
        noun_logits = torch.mm(noun_emb.view(IMG_NUM * OBJECT_LEN, -1), all_noun_embs.t())
        role_mask_ = torch.index_select(input=role_masks, dim=0, index=y_verb_)  # batch * role_num
        # print('y_verb_', y_verb_)
        # for _ in y_verb_:
        #     print('role_masks', role_masks[_])
        # print('role_mask_', role_mask_)

        if bbox_entities_id is None:
            noun_logits = F.log_softmax(noun_logits, dim=-1).view(IMG_NUM, OBJECT_LEN, -1)  # batch * role_num * noun_num
            y_role_ = torch.max(noun_logits, dim=-1)[1]  # batch * role_num
            y_role_ = y_role_.masked_fill(role_mask_.eq(0), 0)  # role['other'] = 0
            heatmap_all.append(heatmap)
        else:
            role_logits = self.sr_model.get_role_logits(noun_emb)
            # print('role_logits_raw', role_logits.size(), role_logits[0][0])
            role_mask_ = role_mask_.float()
            for i in range(IMG_NUM):
                s_len = int(object_num_batch[batch_id][i])
                role_logits[i, s_len:] = 1e-45
                # noun_logits[i, s_len:] = 1e-45
                # event_ae_logits[i, s_len:] = 1e-45
                role_logits[i] = role_logits[i] + (role_mask_[i] + 1e-45).log()
            # print('role_logits', role_logits.size(),  role_logits[0][0])
            role_logits = F.log_softmax(role_logits, dim=-1).view(IMG_NUM, OBJECT_LEN, -1)
            # print('role_logits_log', role_logits.size(), role_logits[0][0])

            # noun_logits = F.log_softmax(noun_logits, dim=-1).view(IMG_NUM, OBJECT_LEN, -1)
            y_role_ = torch.max(role_logits, dim=-1)[1]  # batch * obj_num
            # print('y_role_', y_role_)

        y_role_all_.append(y_role_)
        image_id_all.append(img_id_batch[batch_id][:IMG_NUM])
        doc_image_result[docid] = (
        verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, image_id_all)

        return verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, heatmap_all, image_id_all

    def predict_image_only(self,
                img_id_batch, image_batch,
                role_masks,
                add_object=False, bbox_entities_id=None, bbox_entities_region=None,
                bbox_entities_label=None, object_num_batch=None):
        # get the image embedding, verb_logits, noun_logits, attention_matrix?
        # directly use the mapping
        BATCH_SIZE = image_batch.size(0)
        # IMG_MAX_NUM = image_batch.size(1)
        if not add_object:
            OBJECT_LEN = self.sr_model.get_role_num()
            SEQ_LEN = OBJECT_LEN + 1
        else:
            OBJECT_LEN = bbox_entities_region.size()[-4]
            SEQ_LEN = OBJECT_LEN + 1  # the num of the GCN nodes

        all_verb_embs = self.sr_model.wvembeddings(
            torch.arange(self.sr_model.hyperparams['wvemb_size']).to(self.sr_model.device))
        all_noun_embs = self.sr_model.wnembeddings(
            torch.arange(self.sr_model.hyperparams['wnemb_size']).to(self.sr_model.device))

        y_verb_all_ = []
        y_role_all_ = []
        heatmap_all = []
        image_id_all = []
        doc_image_result = dict()
        for batch_id in range(BATCH_SIZE):
            # for img_id in range(IMG_LEN):
            # IMG_LEN images are input to the SR model together, just like one batch
            verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, \
            heatmap_all, image_id_all = \
                self.predict_image(doc_image_result, batch_id, img_id_batch, image_batch,
                                   bbox_entities_id, bbox_entities_region, bbox_entities_label,
                                   object_num_batch, OBJECT_LEN, SEQ_LEN,
                                   all_verb_embs, all_noun_embs, role_masks,
                                   y_verb_all_, y_role_all_, heatmap_all, image_id_all)

        return y_verb_all_, y_role_all_, heatmap_all

    def predict(self, word_sequence, x_len,
                pos_tagging_sequence, entity_type_sequence, adj, batch_golden_entities,
                img_id_batch, image_batch, label_i2s,
                role_masks, ee_role_mask,
                add_object=False, bbox_entities_id=None, bbox_entities_region=None,
                bbox_entities_label=None, object_num_batch=None,
                apply_ee_role_mask=False, sent_id=None, joint_infer=False):
        '''
        joint inference

        one sentence with multiple images
        :return:
        '''

        # get the text embedding (after self-attetion)
        word_common, word_mask, word_emb = self.ed_model.get_common_feature(word_sequence, x_len, pos_tagging_sequence,
                                                                            entity_type_sequence, adj)

        # get the image embedding, verb_logits, noun_logits, attention_matrix?
        # directly use the mapping
        BATCH_SIZE = image_batch.size(0)
        # IMG_MAX_NUM = image_batch.size(1)
        if not add_object:
            OBJECT_LEN = self.sr_model.get_role_num()
            SEQ_LEN = OBJECT_LEN + 1
        else:
            OBJECT_LEN = bbox_entities_region.size()[-4]
            SEQ_LEN = OBJECT_LEN + 1  # the num of the GCN nodes

        all_verb_embs = self.sr_model.wvembeddings(
            torch.arange(self.sr_model.hyperparams['wvemb_size']).to(self.sr_model.device))
        all_noun_embs = self.sr_model.wnembeddings(
            torch.arange(self.sr_model.hyperparams['wnemb_size']).to(self.sr_model.device))

        # max_img_idx_tensor = torch.zeros.
        max_image_commons = []
        max_grounding_scores = []
        # grounding_score_all = torch.zeros(SENT_NUM, IMG_NUM)
        # list of list:
        y_verb_all_ = []
        y_role_all_ = []
        heatmap_all = []
        image_id_all = []
        doc_image_result = dict()
        # doc_id_batch = list()
        for batch_id in range(BATCH_SIZE):
            # image_id_first = img_id_batch[batch_id][0]
            # docid = image_id_first[:image_id_first.rfind('_')]
            # doc_id_batch.append(docid)

            # for img_id in range(IMG_LEN):
            # IMG_LEN images are input to the SR model together, just like one batch
            verb_emb_common, noun_emb_common, verb_emb, noun_emb, y_verb_all_, y_role_all_, \
                heatmap_all, image_id_all = \
                self.predict_image(doc_image_result, batch_id, img_id_batch, image_batch,
                              bbox_entities_id, bbox_entities_region, bbox_entities_label,
                              object_num_batch, OBJECT_LEN, SEQ_LEN,
                              all_verb_embs, all_noun_embs, role_masks,
                              y_verb_all_, y_role_all_, heatmap_all, image_id_all)

            # get the most similar image for the sentence
            # # sentence copy multiple times?
            image_common_g, sent_common_g, word2noun_att_output_g, word_common_g, noun2word_att_output_g, noun_emb_g = self.similarity(verb_emb_common,
                                                                noun_emb_common, verb_emb, noun_emb,
                                                                word_common[batch_id],  #torch+一个list复制n遍&oq=torch+一个list复制n遍???
                                                                word_emb[batch_id],
                                                                word_mask[batch_id])
            grounding_score = torch.mm(sent_common_g, image_common_g.t()) # [img_num, img_num]

            max_grounding_score, max_img_idx = torch.max(grounding_score, dim=-1, keepdim=False) #[1,]  # select one from all sentence-variance contexted by diff images
            # max_img_idx = torch.argmax(grounding_score, dim=-1, keepdim=True)
            max_grounding_scores.append(max_grounding_score)  #[1,]
            max_image_commons.append(image_common_g[max_img_idx])  #[1, dim]
        # max_grounding_score = grounding_score[:, :, max_img_idx_tensor, :] #torch.index_select(grounding_score, max_img_idx_tensor, dim=1)
        # max_image_common = image_common[:, max_img_idx_tensor, :]  #torch.index_select(max_image_common, image_common[max_img_idx], dim=1)
        max_grounding_scores = torch.stack(max_grounding_scores)  #[batch,]
        max_image_commons = torch.stack(max_image_commons)  #[batch, dim]


        if joint_infer:
            combined_vec = torch.add(word_common, 0.01 * max_grounding_scores.unsqueeze(1) * max_image_commons) #/ 2.0 # (batch_size, seq_len, d')
        else:
            combined_vec = word_common

        # input them to the classifier (classifier is from ee_model)
        # text prediction
        # print('combined_vec', combined_vec.size())
        # print(self.ed_model)
        # logits = self.ed_model.ol(combined_vec)  # (batch_size, seq_len, d') -> (batch_size, seq_len, verb_logits)
        logits = self.ed_model.ace_classifier.forward_type(combined_vec)
        # print('word_common', word_common.size())
        # print('logits', logits.size())

        if apply_ee_role_mask:
            typestr2id, role_mask_matrix = ee_role_mask
            trigger_labels_predicted = []

        ae_logits_key = []
        ae_hidden = []
        trigger_outputs = torch.max(logits, 2)[1].view(logits.size()[:2])  # (batch_size, seq_len)
        predicted_event_triggers_batch = []

        for i in range(BATCH_SIZE):
            # append '__' to the trigger start
            predicted_event_triggers = EDTester.merge_segments(
                [label_i2s[x] for x in trigger_outputs[i][:x_len[i]].tolist()], sent_id[i])
            predicted_event_triggers_batch.append(predicted_event_triggers)

            golden_entities = batch_golden_entities[i]
            golden_entity_tensors = {}
            for j in range(len(golden_entities)):
                e_st, e_ed, e_type_str = golden_entities[j]
                try:
                    golden_entity_tensors[golden_entities[j]] = combined_vec[i, e_st:e_ed, ].mean(dim=0)  # (d')
                except:
                    print(combined_vec.size())
                    print(e_st, e_ed)
                    print(combined_vec[i, e_st:e_ed, ].mean(dim=0).size())
                    exit(-1)

            for sentid_st in predicted_event_triggers:
                ed, trigger_type_str = predicted_event_triggers[sentid_st]
                st = int(sentid_st[sentid_st.find('__')+2:])
                # print('ed', ed)  # ed=ed
                # print('trigger_type_str', trigger_type_str)  # CONFLICT||ATTACK
                event_tensor = combined_vec[i, st:ed, ].mean(dim=0)  # (d')
                for j in range(len(golden_entities)):
                    if apply_ee_role_mask:
                        trigger_labels_predicted.append(typestr2id[trigger_type_str])
                    e_st, e_ed, e_type_str = golden_entities[j]
                    entity_tensor = golden_entity_tensors[golden_entities[j]]
                    ae_hidden.append(torch.cat([event_tensor, entity_tensor]))  # (2 * d')
                    ae_logits_key.append((i, sentid_st, ed, trigger_type_str, e_st, e_ed, e_type_str))
        if len(ae_hidden) != 0:
            # ae_logits = self.ed_model.ae_ol(torch.stack(ae_hidden, dim=0))
            ae_logits = self.ed_model.ace_classifier.forward_role(torch.stack(ae_hidden, dim=0))
            # ae_input = torch.stack(ae_hidden, dim=0)
            # ae_hidden = self.ae_bn1(F.relu(self.ae_l1(ae_input)))
            # ae_hidden = self.ae_l2(ae_hidden)

            # mask logits of roles
            if apply_ee_role_mask:
                trigger_labels_predicted = torch.LongTensor(trigger_labels_predicted).to(self.device)
                role_matrix = torch.index_select(input=role_mask_matrix, dim=0, index=trigger_labels_predicted)
                ae_logits = ae_logits.masked_fill(role_matrix.eq(0), float('-inf'))

            predicted_events = [{} for _ in range(BATCH_SIZE)]
            output_ae = torch.max(ae_logits, dim=-1)[1].tolist()  #.view(golden_labels.size()).tolist()
            for (i, sentid_st, ed, event_type_str, e_st, e_ed, entity_type), ae_label in zip(ae_logits_key, output_ae):
                if ae_label == consts.ROLE_O_LABEL:
                    # print('ae_label is OTHER')
                    continue
                if (sentid_st, ed, event_type_str) not in predicted_events[i]:
                    predicted_events[i][(sentid_st, ed, event_type_str)] = []
                predicted_events[i][(sentid_st, ed, event_type_str)].append((e_st, e_ed, ae_label))
        else:
            predicted_events = [{} for _ in range(len(x_len))]

        if joint_infer:
            # image added text feature to inference
            y_verb_all_joint_ = []
            y_role_all_joint_ = []
            for batch_id in range(BATCH_SIZE):
                image_id_first = img_id_batch[batch_id][0]
                docid = image_id_first[:image_id_first.rfind('_')]
                IMG_NUM = len(img_id_batch[batch_id])

                if max_grounding_scores[batch_id] > 1.0:
                    verb_emb_common_joint = torch.add(verb_emb_common, (
                                0.01 * max_grounding_scores[batch_id] * word_common))  # / 2.0 # (batch_size, seq_len, d')

                _, __, ___, ____, y_verb_all_joint_, y_role_all_joint_, heatmap_all_joint, \
                image_id_all = self.image_inference(self, verb_emb_common_joint, noun_emb_common, verb_emb, noun_emb, heatmap,
                                doc_image_result, batch_id, img_id_batch, image_batch,
                                bbox_entities_id, bbox_entities_region, bbox_entities_label,
                                object_num_batch, OBJECT_LEN, SEQ_LEN,
                                all_verb_embs, all_noun_embs, role_masks,
                                y_verb_all_joint_, y_role_all_joint_, heatmap_all, image_id_all,
                                IMG_NUM, docid)

                print(y_verb_all_, y_role_all_, heatmap_all, y_verb_all_joint_, y_role_all_joint_, heatmap_all_joint)

        return logits, predicted_event_triggers_batch, predicted_events, y_verb_all_, y_role_all_, heatmap_all, max_grounding_scores