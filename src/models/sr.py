import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules.DynamicLSTM import DynamicLSTM
from src.models.modules.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer
from src.models.modules.FMapLayerImage import FMapLayerImage
from src.models.modules.GCN import GraphConvolution
from src.models.modules.HighWay import HighWay
from src.models.modules.model import Model
from src.util.util_model import BottledXavierLinear, BottledMLP, masked_log_softmax
from src.util import consts
# from src.models.ee import calculate_loss_ed, calculate_loss_ae

class SRModel(Model):
    def __init__(self, hyps,
                 embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                 device, ace_classifier):
        super(SRModel, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device

        assert embeddingMatrix_noun is not None
        assert embeddingMatrix_verb is not None
        assert embeddingMatrix_role is not None

        # Image Embedding Layer
        self.ivembeddings = FMapLayerImage(fine_tune=hyps["iemb_ft"],
                                                dropout=hyps["iemb_dp"],
                                                backbone=hyps['iemb_backbone'],
                                                device=device)
        self.inembeddings = FMapLayerImage(fine_tune=hyps["iemb_ft"],
                                                dropout=hyps["iemb_dp"],
                                                backbone=hyps['iemb_backbone'],
                                                device=device)
        # print('device', device, 'iembeddings', self.iembeddings)

        # Noun Embedding Layer
        self.wnembeddings = EmbeddingLayer(embedding_size=(hyps["wnemb_size"], hyps["wemb_dim"]),
                                           embedding_matrix=embeddingMatrix_noun,
                                           fine_tune=hyps["wemb_ft"],
                                           dropout=hyps["wemb_dp"],
                                           device=device)
        # Verb Embedding Layer
        self.wvembeddings = EmbeddingLayer(embedding_size=(hyps["wvemb_size"], hyps["wemb_dim"]),
                                            embedding_matrix=embeddingMatrix_verb,
                                            fine_tune=hyps["wemb_ft"],
                                            dropout=hyps["wemb_dp"],
                                            device=device)
        # Role Embedding Layer
        self.wrembeddings = EmbeddingLayer(embedding_size=(hyps["wremb_size"], hyps["wemb_dim"]),
                                            embedding_matrix=embeddingMatrix_role,
                                            fine_tune=hyps["wemb_ft"],
                                            dropout=hyps["wemb_dp"],
                                            device=device)

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(hyps["gcn_layers"]):
            gcn = GraphConvolution(in_features=hyps["wemb_dim"],
                                   out_features=hyps["wemb_dim"],
                                   edge_types=2,
                                   dropout=hyps["gcn_dp"] if i != hyps["gcn_layers"] - 1 else None,
                                   use_bn=hyps["gcn_use_bn"],
                                   device=device)
            self.gcns.append(gcn)
        
        self.emb2feat_verb = BottledXavierLinear(in_features=hyps["wemb_dim"], out_features=hyps["iemb_dim"]).to(device)
        self.emb2feat_role = BottledXavierLinear(in_features=hyps["wemb_dim"], out_features=hyps["iemb_dim"]).to(device)
        self.emb2feat_pos = BottledXavierLinear(in_features=hyps["posemb_dim"], out_features=hyps["fmap_dim"]).to(device)
                                                                                                                
        self.feat2emb_noun = BottledMLP([hyps["iemb_dim"], hyps["iemb_dim"], hyps["wemb_dim"]]).to(device)
        self.feat2emb_verb = BottledMLP([hyps["iemb_dim"], hyps["iemb_dim"], hyps["wemb_dim"]]).to(device)

        self.att_key_head = BottledMLP([hyps["fmap_dim"], hyps["att_dim"]], act_fn='Tanh', last_act=True).to(device)
        self.att_query_head = BottledMLP([hyps["iemb_dim"], hyps["att_dim"]], act_fn='Tanh', last_act=True).to(device)
        
        self.fmap_pool = BottledXavierLinear(hyps["fmap_dim"], hyps["iemb_dim"]).to(device)
        
        
        posemb = np.random.randn(hyps['fmap_size'], hyps['fmap_size'], hyps["posemb_dim"])
        posemb = np.maximum(np.minimum(posemb, 1.0), -1.0).astype(np.float32)
        self.position_embedding = torch.tensor(posemb, device='cuda', requires_grad=True)

        self.role_num = hyps["wremb_size"]
        print('role_num', self.role_num)
        self.common_dim = hyps["wemb_dim"]  # ??? from feat2emb_noun

        self.ace_classifier = ace_classifier
        # # Output Linear
        # self.ol = BottledXavierLinear(in_features=self.common_dim, out_features=hyps["oc"]).to(device=device)
        #
        # # AE Output Linear
        # self.ae_ol = BottledXavierLinear(in_features=2 * self.common_dim, out_features=hyps["ae_oc"]).to(device=device)
        
        # Move to right device
        self.to(self.device)

    def get_common_dim(self):
        return self.common_dim

    def get_role_num(self):
        return self.role_num

    def get_common_feature(self, img_id_batch, image_batch,
                           bbox_entities_id, bbox_entities_region, bbox_entities_label,
                           object_num_batch, BATCH_SIZE, OBJECT_LEN, SEQ_LEN):
        '''
        Get multimedia common semantic representation
        :param img_id_batch: list, [batch,]
        :param image_batch: [batch, 3, 200, 200]
        :param bbox_entities_id: list(list)
        :param bbox_entities_region: [batch, object_num, 3, 200, 200]
        :param bbox_entities_label: [batch, object_num]
        :param object_num_batch: narray, [batch, ]
        :param BATCH_SIZE:
        :param OBJTECT_LEN:
        :param SEQ_LEN:
        :return:
        '''

        # Embedding image
        _, verb_feats = self.ivembeddings(image_batch)  # batchsize * 2048 * imageLength * imageLength (imagelength = 1 or 7)
        verb_feats = verb_feats.view(BATCH_SIZE, -1)  # batchsize * (2048*imageLength_sqare)

        noun_fmap, noun_feats = self.inembeddings(image_batch)  # batchsize * 2048 * imageLength * imageLength (imagelength = 1 or 7)
        noun_feats = noun_feats.view(BATCH_SIZE, -1)  # batchsize * (2048*imageLength_sqare)

        noun_fmap = noun_fmap.permute(0, 2, 3, 1)
                
        all_role_embs = self.wrembeddings(torch.arange(self.role_num).to(self.device))
        # arg_feats = torch.cat([
        #     noun_feats.unsqueeze(1).expand(BATCH_SIZE, self.role_num, -1),
        #     self.emb2feat_role(all_role_embs).unsqueeze(0).expand(BATCH_SIZE, self.role_num, -1),
        # ], dim=-1)
        arg_feats = torch.add(
            noun_feats.unsqueeze(1).expand(BATCH_SIZE, 192, -1),
            self.emb2feat_role(all_role_embs).unsqueeze(0).expand(BATCH_SIZE, 192, -1),
        )

        noun_fmap = torch.add(
            noun_fmap, 
            self.emb2feat_pos(self.position_embedding).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
        )
        
        att_query = self.att_query_head(arg_feats)  # batch, role_num, att_dim
        att_key = self.att_key_head(noun_fmap).view(BATCH_SIZE, -1, self.hyperparams['att_dim']) # batch, 49, att_dim
        # print('att_query', att_query)
        # print('att_key', att_key)
        att_weights = F.softmax(torch.bmm(att_query, att_key.permute(0, 2, 1)) / att_key.size(1), dim=-1) # torch.matmul
        # print('att_weights', att_weights)
        heatmap = att_weights.view(BATCH_SIZE, self.role_num, noun_fmap.size(1), noun_fmap.size(2))
        
        att_feat = torch.sum(heatmap.unsqueeze(-1) * self.fmap_pool(noun_fmap).unsqueeze(1), dim=[2, 3])
        
        arg_feats = torch.add(
            att_feat, 
            self.emb2feat_role(all_role_embs).unsqueeze(0).expand(BATCH_SIZE, self.role_num, -1),
        )

        noun_emb = self.feat2emb_noun(arg_feats)
        verb_emb = self.feat2emb_verb(verb_feats)

        noun_emb_common = noun_emb
        verb_emb_common = verb_emb
        
        adj = np.zeros((BATCH_SIZE, 2, self.role_num + 1, self.role_num + 1), dtype='float32')
        adj[:, 0, 0, 1:] = 1.0
        adj[:, 1, 1:, 0] = 1.0
        adj = torch.from_numpy(adj).to(self.device)
        
        for i in range(self.hyperparams["gcn_layers"]):
            nodes = torch.cat([verb_emb_common.unsqueeze(1), noun_emb_common], axis=1)
            nodes = self.gcns[i](nodes, adj)
        
            verb_emb_residue = nodes[:, 0, :].squeeze(1)
            noun_emb_residue = nodes[:, 1:, :]

            verb_emb_common = verb_emb_common + verb_emb_residue
            noun_emb_common = noun_emb_common + noun_emb_residue

        # print('verb_emb_common', verb_emb_common.size())
        # print('noun_emb_common', noun_emb_common.size())
        # print('verb_emb', verb_emb.size())
        # print('noun_emb', noun_emb.size())
        # print('heatmap_origion0', heatmap)
        return verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap

    def forward(self, img_id_batch, image_batch,
                bbox_entities_id, bbox_entities_region, bbox_entities_label, bbox_num_batch):
        '''

        :param img_id_batch: list, [batch,]
        :param image_batch: [batch, 3, 200, 200]
        :param bbox_entities_id: list(list), [batch, obj_num]
        :param bbox_entities_region: [batch, object_num, 3, 200, 200]
        :param bbox_entities_label: [batch, object_num]
        :param bbox_num_batch: narray, [batch, ]
        :return:
        '''
        BATCH_SIZE = image_batch.size()[0]  # word_sequence.size()[0]
        # if bbox_entities_region is None:
        OBJECT_LEN = self.role_num
        SEQ_LEN = OBJECT_LEN + 1
        # else:
        #     OBJTECT_LEN = bbox_entities_region.size()[1]
        #     SEQ_LEN = OBJTECT_LEN + 1

        entity_num_batch = torch.LongTensor(np.ones((BATCH_SIZE)) * self.role_num).to(self.device)
        # entity_mask = torch.ByteTensor(np.zeros((BATCH_SIZE, self.role_num))).to(self.device)

        verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = self.get_common_feature(img_id_batch, image_batch,
                           bbox_entities_id, bbox_entities_region, bbox_entities_label,
                           bbox_num_batch, BATCH_SIZE, OBJECT_LEN, SEQ_LEN)

        role_logits = torch.eye(self.role_num).unsqueeze(0).expand(BATCH_SIZE, -1, -1).to(self.device)

        # get logits of verb
        all_verb_embs = self.wvembeddings(torch.arange(self.hyperparams['wvemb_size']).to(self.device))
        verb_logits = torch.mm(verb_emb, all_verb_embs.t())
        verb_logits = F.log_softmax(verb_logits, dim=-1)
        # get logits of noun
        OBJ_NUM = noun_emb.size(1)
        all_noun_embs = self.wnembeddings(torch.arange(self.hyperparams['wnemb_size']).to(self.device))
        noun_logits = torch.mm(noun_emb.view(BATCH_SIZE * OBJ_NUM, -1), all_noun_embs.t())
        noun_logits = F.log_softmax(noun_logits, dim=-1).view(BATCH_SIZE, OBJ_NUM, -1)
        # ! no need to mask the object level, as obj_num is the same for all batches, i.e., equals to 192

        # addtional classifier
        event_logits, event_ae_logits = self.ace_event_role_extract(verb_emb_common, noun_emb_common, verb_logits, noun_logits)
        # event_logits, event_ae_logits = None, None

        return img_id_batch, verb_emb_common, noun_emb_common, verb_emb, noun_emb, \
               role_logits, verb_logits, noun_logits, event_logits, event_ae_logits, \
               entity_num_batch #, entity_mask


    def ace_event_role_extract(self, verb_emb_common, noun_emb_common, verb_logits, noun_logits):

        # event_logits = self.ol(verb_emb_common)
        event_logits = self.ace_classifier.forward_type(verb_emb_common)

        # ae_logits_key = []
        # ae_hidden = []
        # trigger_outputs = torch.max(verb_logits, dim=1)[1]
        event_tensor = verb_emb_common.unsqueeze(1).expand_as(noun_emb_common)  # batch, 192, dim

        # noun_outputs = torch.max(noun_logits, dim=2)[1]  # batch * role_num
        # role_mask_ = torch.index_select(input=role_masks, dim=0, index=trigger_outputs)  # batch * role_num
        # y_role_ = noun_outputs.masked_fill(role_mask_.eq(0), 0)  # role['other'] = 0

        ae_hidden = torch.cat([event_tensor, noun_emb_common], dim=2)

        # event_ae_logits = self.ae_ol(ae_hidden)
        event_ae_logits = self.ace_classifier.forward_role(ae_hidden)
        # ae_input = torch.stack(ae_hidden, dim=0)
        # ae_hidden = self.ae_bn1(F.relu(self.ae_l1(ae_input)))
        # ae_hidden = self.ae_l2(ae_hidden)

        # # mask argument logits
        # event_ae_logits = masked_log_softmax(event_ae_logits, ent_mask, dim=-1)

        return event_logits, event_ae_logits


    def calculate_loss_ace(self, event_logits, event_ae_logits, event_gt_batch, ee_roles_gt_batch, num_entity_proj, num_entity_gt):
        # calculate_loss_ed(self, logits, mask, label, weight)
        BATCH_SIZE = event_logits.size()[0]
        # if weight is not None:
        #     weight = weight.to(self.device)
        #     cost_ed = F.nll_loss(F.log_softmax(event_logits, dim=1), event_gt_batch, weight=weight)
        # else:
        cost_ed = F.nll_loss(F.log_softmax(event_logits, dim=1), event_gt_batch)

        num_entity_gt = torch.LongTensor(num_entity_gt).to(self.device)

        # loss of noun embedding projection
        event_ae_logits = F.log_softmax(event_ae_logits, dim=2)
        cost_ae = 0.0
        for i in range(BATCH_SIZE):
            oi = ee_roles_gt_batch[i, :num_entity_gt[i]]  # ground truth role idx list
            # prerequistes: role_idx = entity_idx, for each role, we predict one entity, so there are 192 entities
            # get role index == get entity entity idx, so can be used to select the values from event_ae_logits (batch, obj_num, role_num)
            ent_cost = torch.sum(
                - event_ae_logits[i, oi][torch.arange(num_entity_gt[i]), ee_roles_gt_batch[i, :num_entity_gt[i]]])
            cost_ae = cost_ae + ent_cost
        cost_ae = cost_ae / BATCH_SIZE

        loss_ace_terms = {'ace_ed': cost_ed.item(), 'ace_ae': cost_ae.item()}
        loss_sr_ace = (self.hyperparams['loss_weight_verb'] * cost_ed) + (
                    self.hyperparams['loss_weight_noun'] * cost_ae)

        return loss_sr_ace, loss_ace_terms, event_logits, event_ae_logits


    def calculate_loss_all(self, emb_verb_proj, idx_verb_gt, emb_obj_proj, idx_entity_gt, role_logits, verb_logits, noun_logits,
                           role_gt_type, num_entity_proj, num_entity_gt):
        '''
        :param emb_verb_proj: size = [batch_size, ]
        :param emb_verb_gt: size = [batch_size, ]
        :param emb_proj: size = [batch_size, OBJ_NUM, emb_dim]
        :param emb_entity_gt: size = [batch_size, ARG_NUM, emb_dim]
        :param role_logits: size = [batch_size, OBJ_NUM, num_role_types]
        :param role_gt_type: size = [batch_size, ARG_NUM]
        :return:
        '''
        #ROLE_NUM = role_logits.size(2)
        BATCH_SIZE = emb_obj_proj.size(0)
        # OBJ_NUM = emb_obj_proj.size(1)
        # ARG_NUM = role_gt_type.size(1)
        
        num_entity_gt = torch.LongTensor(num_entity_gt).to(self.device)
        
        # loss of verb embedding projection
        cost_verb = torch.mean(- verb_logits[torch.arange(BATCH_SIZE), idx_verb_gt])

        # loss of noun embedding projection
        cost_entity_emb = 0.0
        for i in range(BATCH_SIZE):
            oi = role_gt_type[i, :num_entity_gt[i]]
            ent_cost = torch.sum(- noun_logits[i, oi][torch.arange(num_entity_gt[i]), idx_entity_gt[i, :num_entity_gt[i]] ])
            cost_entity_emb = cost_entity_emb + ent_cost
        cost_entity_emb = cost_entity_emb / BATCH_SIZE

        loss_terms = {'verb': cost_verb.item(), 'noun': cost_entity_emb.item()}
        loss_sr = (self.hyperparams['loss_weight_verb'] * cost_verb) + (self.hyperparams['loss_weight_noun'] * cost_entity_emb)

        return loss_sr, loss_terms, verb_logits, noun_logits, role_logits

    # def get_all_verbs_emb(self):
    #     # get all indexes, should include <PAD>
    #     verb_idxes = torch.from_numpy(np.arange(self.hyperparams['wvemb_size'])).to(self.device) # [0,1,....,verb_size-1],
    #     embed_verb_all = self.wvembeddings(verb_idxes)  # batch, verb_size, embed_verb_dim
    #     return embed_verb_all
    #
    # def get_predicted_event(self, emb_verb_proj, emb_verb_all):
    #     BATCH_SIZE = emb_verb_proj.size(0)
    #     VERB_SIZE = emb_verb_all.size(0)
    #     EMB_SIZE = emb_verb_all.size(1)
    #     emb_verb_all_ = emb_verb_all.unsqueeze(0).expand(BATCH_SIZE, VERB_SIZE, EMB_SIZE)
    #     emb_verb_proj_ = emb_verb_proj.unsqueeze(1).expand(BATCH_SIZE, VERB_SIZE, EMB_SIZE) # batch, 1, embed_verb_dim
    #     # dist = F.mse_loss(emb_verb_proj_, embed_verb_all, size_average=False)  # batch, verb_size
    #     dist = (emb_verb_proj_ - emb_verb_all_).pow(2).mean(dim=-1)
    #     min_verb_value, min_verb_idx = torch.min(dist, dim=1)
    #     return min_verb_idx  # batch,
    #
    # def get_all_nouns_emb(self):
    #     # get all indexes, should include <PAD>
    #     noun_idxes = torch.from_numpy(np.arange(self.hyperparams['wnemb_size'])).to(self.device) # [0,1,....,verb_size-1],
    #     embed_noun_all = self.wnembeddings(noun_idxes)  # batch, verb_size, embed_verb_dim
    #     return embed_noun_all
    #
    # def get_predicted_entities(self, emb_noun_proj, emb_noun_all):
    #     '''
    #     get predicted nouns for objects in a batch
    #     :param emb_noun_proj: [batch, obj_num, emb_dim]
    #     :param emb_noun_all: [nounvocab_size, emb_dim]
    #     :return:
    #     '''
    #     BATCH_SIZE = emb_noun_proj.size(0)
    #     OBJ_NUM = emb_noun_proj.size(1)
    #     NOUN_SIZE = emb_noun_all.size(0)
    #     EMB_SIZE = emb_noun_all.size(1)
    #     emb_noun_all_ = emb_noun_all.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, OBJ_NUM, NOUN_SIZE, EMB_SIZE)
    #     emb_noun_proj_ = emb_noun_proj.unsqueeze(2).expand(BATCH_SIZE, OBJ_NUM, NOUN_SIZE, EMB_SIZE) # batch, 1, embed_verb_dim
    #     dist = (emb_noun_proj_ - emb_noun_all_).pow(2).mean(dim=-1)
    #     min_noun_value, min_noun_idx = torch.min(dist, dim=2)
    #     return min_noun_idx  # [batch, obj_num]
    #
    # def get_predicted_entities_each_batch(self, emb_noun_proj, emb_noun_all):
    #     '''
    #     get predicted nouns for objects in a batch
    #     :param emb_noun_proj: [obj_num, emb_dim]
    #     :param emb_noun_all: [nounvocab_size, emb_dim]
    #     :return:
    #     '''
    #     # emb_noun_all = emb_noun_all.cpu()
    #     # emb_noun_proj = emb_noun_proj.cpu()
    #     OBJ_NUM = emb_noun_proj.size(0)
    #     NOUN_SIZE = emb_noun_all.size(0)
    #     EMB_SIZE = emb_noun_all.size(1)
    #     # emb_noun_all_ = emb_noun_all.unsqueeze(0).expand(OBJ_NUM, NOUN_SIZE, EMB_SIZE).cpu()
    #     # emb_noun_proj_ = emb_noun_proj.unsqueeze(1).expand(OBJ_NUM, NOUN_SIZE, EMB_SIZE).cpu() # batch, 1, embed_verb_dim
    #     # dist = (emb_noun_proj_ - emb_noun_all_).pow(2).mean(dim=-1)
    #     # min_noun_value, min_noun_idx = torch.min(dist, dim=1)
    #     min_noun_idx = []
    #     for obj_idx in range(OBJ_NUM):
    #         # print('obj_idx', obj_idx)
    #         # use matrix:
    #         emb_noun_proj_ = emb_noun_proj[obj_idx].unsqueeze(0).expand(NOUN_SIZE, EMB_SIZE)  # [nounvocab_size, emb_dim]
    #         dist = (emb_noun_proj_ - emb_noun_all).pow(2).mean(dim=-1)  # [nounvocab_size,]
    #         # print('dist', dist.size())
    #         min_value, min_idx = torch.min(dist, dim=0)
    #         # print('min_idx', min_idx.item())
    #         min_noun_idx.append(min_idx.item())
    #         # # use loops:
    #         # dists = []
    #         # for noun_idx in range(NOUN_SIZE):
    #         #     dist = (emb_noun_proj[obj_idx] - emb_noun_all[noun_idx]).pow(2).mean(dim=-1).item()
    #         #     dists.append(dist)
    #         # print('dists', dists)
    #         # min_idx = dists.index(min(dists))
    #         # print('min_idx', min_idx)
    #         # min_noun_idx.append(min_idx)
    #         # print('min_noun_idx', min_noun_idx)
    #     return min_noun_idx  # [obj_num]


    # def calculate_loss_role(self, logits, batch_golden_events, BATCH_SIZE):
    #     golden_labels = []
    #     for i, st, ed, event_type_str, e_st, e_ed, entity_type in keys:
    #         label = consts.ROLE_O_LABEL
    #         if (st, ed, event_type_str) in batch_golden_events[i]:  # if event matched
    #             for e_st_, e_ed_, r_label in batch_golden_events[i][(st, ed, event_type_str)]:
    #                 if e_st == e_st_ and e_ed == e_ed_:
    #                     label = r_label
    #                     break
    #         golden_labels.append(label)
    #     golden_labels = torch.LongTensor(golden_labels).to(self.device)
    #     loss = F.nll_loss(F.log_softmax(logits, dim=1), golden_labels)
    #
    #     predicted_events = [{} for _ in range(BATCH_SIZE)]
    #     output_ae = torch.max(logits, 1)[1].view(golden_labels.size()).tolist()
    #     for (i, st, ed, event_type_str, e_st, e_ed, entity_type), ae_label in zip(keys, output_ae):
    #         if ae_label == consts.ROLE_O_LABEL: continue
    #         if (st, ed, event_type_str) not in predicted_events[i]:
    #             predicted_events[i][(st, ed, event_type_str)] = []
    #         predicted_events[i][(st, ed, event_type_str)].append((e_st, e_ed, ae_label))
    #
    #     return loss, predicted_events