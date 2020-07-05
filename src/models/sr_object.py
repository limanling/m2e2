import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules.DynamicLSTM import DynamicLSTM
from src.models.modules.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer
from src.models.modules.EmbeddingLayerImage import EmbeddingLayerImage
from src.models.modules.FMapLayerImage import FMapLayerImage
from src.models.modules.GCN import GraphConvolution
from src.models.modules.HighWay import HighWay
from src.models.modules.model import Model
from src.models.sr import SRModel
from src.util.util_model import BottledXavierLinear, BottledMLP, masked_log_softmax
from src.util import consts


class SRModel_Object(Model):
    def __init__(self, hyps,
                 embeddingMatrix_noun, embeddingMatrix_verb, embeddingMatrix_role,
                 device, ace_classifier):
        super(SRModel_Object, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device

        assert embeddingMatrix_noun is not None
        assert embeddingMatrix_verb is not None
        assert embeddingMatrix_role is not None

        # Image Embedding Layer
        # self.iembeddings = EmbeddingLayerImage(fine_tune=hyps["iemb_ft"],
        #                                        dropout=hyps["iemb_dp"],
        #                                        device=device)
        self.ivembeddings = FMapLayerImage(fine_tune=hyps["iemb_ft"],
                                           dropout=hyps["iemb_dp"],
                                           backbone=hyps['iemb_backbone'],
                                           device=device)
        self.inembeddings = FMapLayerImage(fine_tune=hyps["iemb_ft"],
                                           dropout=hyps["iemb_dp"],
                                           backbone=hyps['iemb_backbone'],
                                           device=device)

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

        # # Entity Label Embedding Layer (Object Detection Labels)
        # self.eembeddings = MultiLabelEmbeddingLayer(embedding_size=(hyps["eemb_size"], hyps["eemb_dim"]),
        #                                             dropout=hyps["eemb_dp"],
        #                                             device=device)

        # Bi-LSTM Encoder
        # self.bilstm = DynamicLSTM(input_size=hyps["iemb_dim"],
        #                           # + hyps["eemb_dim"],
        #                           hidden_size=hyps["lstm_dim"],
        #                           num_layers=hyps["lstm_layers"],
        #                           dropout=hyps["lstm_dp"],
        #                           bidirectional=True,
        #                           device=device)

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

        # Highway
        if hyps["use_highway"]:
            self.hws = nn.ModuleList()
            for i in range(hyps["gcn_layers"]):
                hw = HighWay(size=hyps["wemb_dim"], dropout_ratio=hyps["gcn_dp"]).to(device)
                self.hws.append(hw)  # Bi-LSTM Encoder

        # Bi-LSTM projection
        # self.proj = DynamicLSTM(input_size=2*hyps["lstm_dim"],
        #                         hidden_size=hyps["wemb_dim"],
        #                         num_layers=hyps["lstm_layers"],
        #                         dropout=hyps["lstm_dp"],
        #                         bidirectional=False,
        #                         device=device)
        self.feat2emb_noun = BottledMLP([hyps["iemb_dim"], hyps["iemb_dim"], hyps["wemb_dim"]]).to(device)
        self.feat2emb_verb = BottledMLP([hyps["iemb_dim"], hyps["iemb_dim"], hyps["wemb_dim"]]).to(device)

        self.role_num = hyps["wremb_size"]
        print('role_num', self.role_num)

        # AE Output Linear (role classifier)
        self.sr_ae_ol = BottledXavierLinear(in_features=hyps["wemb_dim"], out_features=self.role_num).to(device)

        # # proj
        # self.proj_loss = nn.MSELoss(reduction='sum')

        # ace classifier
        self.ace_classifier = ace_classifier

        # Move to right device
        self.to(self.device)

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

        # print('BATCH_SIZE', BATCH_SIZE)
        # print('OBJTECT_LEN', OBJTECT_LEN)
        # print('SEQ_LEN', SEQ_LEN)
        # print('image_batch', image_batch.size())
        # print('input_image_', input_image_.size())

        # Embedding the objects/regions
        # print('in the common ', bbox_entities_region.size())
        bbox_entities_batch_ = bbox_entities_region.view(-1, bbox_entities_region.size()[-3], bbox_entities_region.size()[-2], bbox_entities_region.size()[-1])
        # print('bbox_entities_region', bbox_entities_region.size())
        # print('bbox_entities_batch_', bbox_entities_batch_.size())
        _, input_regions_ = self.inembeddings(bbox_entities_batch_)  # [batchsize*object_num] * 2048 * imageLength * imageLength (imageLength=1)
        input_regions_ = input_regions_.view(BATCH_SIZE, OBJECT_LEN, -1)  # batchsize * object_num * (2048*imageLength_sqare)
        # print('input_regions_', input_regions_.size())
        # ## Embedding the object labels
        # input_regions_label_emb = self.eembeddings(bbox_entities_label) ## does the labels are indexes of a type vocabulary? Or it should use the same word embedding layer?
        # ## get node embeddings
        # bbox_emb = torch.cat([input_image, input_regions_, input_regions_label_emb], 2)  # (batch_size, seq_len, d)

        # # Merge image and objects embeddings
        # x_emb = torch.cat([input_image_, input_regions_], 1)  # batchsize * (1+object_num) * (2048*imageLength_sqare)
        # # print('x_emb', x_emb.size())
        # # x_emb = x_emb.view(BATCH_SIZE, SEQ_LEN, -1)
        # x_len = object_num_batch + 1  # batch_size,
        # # print('object_num_batch', object_num_batch)
        # # print('x_len', x_len)
        # # print('x_emb', x_emb.size())

        # construct the graph
        # adj = torch.zeros(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)  # (batch_size, edge_types, seq_len, seq_len)
        # for i in range(BATCH_SIZE):
        #     for j in range(object_num_batch[i]):
        #         adj[i][0][0][j+1] = 1
        #         adj[i][0][j+1][0] = 1
        #         adj[i][0][j+1][j+1] = 1
        # adj = adj.to(self.device)
        adj = np.zeros((BATCH_SIZE, 2, SEQ_LEN, SEQ_LEN), dtype='float32')
        adj[:, 0, 0, 1:] = 1.0
        adj[:, 1, 1:, 0] = 1.0
        adj = torch.from_numpy(adj).to(self.device)

        # project to word space
        verb_emb = self.feat2emb_verb(verb_feats)
        noun_emb = self.feat2emb_noun(input_regions_)

        noun_emb_common = noun_emb
        verb_emb_common = verb_emb

        nodes_common = torch.cat([verb_emb_common.unsqueeze(1), noun_emb_common], axis=1)
        for i in range(self.hyperparams["gcn_layers"]):
            nodes_residue = self.gcns[i](nodes_common, adj)
            nodes_common = nodes_common + nodes_residue

        heatmap = None

        return verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap  # x, x_len,

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
        BATCH_SIZE = bbox_entities_region.size()[0]  # word_sequence.size()[0]
        OBJECT_LEN = bbox_entities_region.size()[1]
        # if OBJECT_LEN == 0:
        #     OBJECT_LEN = 1
        SEQ_LEN = OBJECT_LEN + 1

        # Create the mask matrix (mask: [1,1,1,0,0]: the padded one is 0)
        # arg_mask = np.zeros(shape=args_batch.size(), dtype=np.uint8)
        # for i in range(BATCH_SIZE):
        #     # for j in range(role_num_batch[i]):
        #     s_len = int(arg_num_batch[i])
        #     arg_mask[i, 0:s_len] = np.ones(shape=(s_len), dtype=np.uint8)
        # arg_mask = torch.ByteTensor(arg_mask).to(self.device)
        bbox_mask = np.zeros(shape=bbox_entities_label.size(), dtype=np.uint8)
        for i in range(BATCH_SIZE):
            s_len = int(bbox_num_batch[i])
            bbox_mask[i, 0:s_len] = np.ones(shape=(s_len), dtype=np.uint8)
        bbox_mask = torch.ByteTensor(bbox_mask).to(self.device)

        verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = self.get_common_feature(img_id_batch, image_batch,
                           bbox_entities_id, bbox_entities_region, bbox_entities_label,
                           bbox_num_batch, BATCH_SIZE, OBJECT_LEN, SEQ_LEN)

        # ## target word embedding
        # embed_verb_gt = self.wvembeddings(verb_batch)  # batch_size, emb_dim
        # # print('args_batch', args_batch.size())
        # embed_entity_gt = self.wnembeddings(args_batch)  # batch_size, entity_len, emb_dim

        # output linear of the objects -> role classifier
        # role_logits = self.sr_ae_ol(torch.cat(verb_emb, noun_emb))  # (batch_size, object_num, sr_ae_oc)
        role_logits = self.get_role_logits(noun_emb)  # role_logits = self.sr_ae_ol(noun_emb)
        # # mask role logits
        # for i in range(BATCH_SIZE):
        #     s_len = int(bbox_num_batch[i])
        #     role_logits[i, s_len:] = 1e-45
        # role_logits = F.log_softmax(role_logits, dim=-1).view(BATCH_SIZE, OBJECT_LEN, -1)
        # # role_logits = masked_log_softmax(role_logits, bbox_mask, dim=-1)

        # get logits of verb
        all_verb_embs = self.wvembeddings(torch.arange(self.hyperparams['wvemb_size']).to(self.device))
        verb_logits = torch.mm(verb_emb, all_verb_embs.t())
        verb_logits = F.log_softmax(verb_logits, dim=-1)
        # get logits of noun
        all_noun_embs = self.wnembeddings(torch.arange(self.hyperparams['wnemb_size']).to(self.device))
        noun_logits = torch.mm(noun_emb.view(BATCH_SIZE * OBJECT_LEN, -1), all_noun_embs.t())
        # for i in range(BATCH_SIZE):
        #     s_len = int(bbox_num_batch[i])
        #     noun_logits[i, s_len:] = 1e-45
        # noun_logits = F.log_softmax(noun_logits, dim=-1).view(BATCH_SIZE, OBJECT_LEN, -1)
        # # noun_logits = masked_log_softmax(noun_logits, bbox_mask.unsqu, dim=-1)

        # addtional classifier
        event_logits, event_ae_logits = self.ace_event_role_extract(verb_emb_common, noun_emb_common, verb_logits,
                                                                    noun_logits)

        # mask and log_softmax
        for i in range(BATCH_SIZE):
            s_len = int(bbox_num_batch[i])
            role_logits[i, s_len:] = 1e-45
            noun_logits[i, s_len:] = 1e-45
            event_ae_logits[i, s_len:] = 1e-45
        role_logits = F.log_softmax(role_logits, dim=-1).view(BATCH_SIZE, OBJECT_LEN, -1)
        noun_logits = F.log_softmax(noun_logits, dim=-1).view(BATCH_SIZE, OBJECT_LEN, -1)
        event_ae_logits = F.log_softmax(event_ae_logits, dim=-1).view(BATCH_SIZE, OBJECT_LEN, -1)
        #   = masked_log_softmax(event_ae_logits, bbox_mask, dim=-1)
        # event_logits, event_ae_logits = None, None

        return img_id_batch, verb_emb_common, noun_emb_common, verb_emb, noun_emb, \
               role_logits, verb_logits, noun_logits, event_logits, event_ae_logits, \
               bbox_num_batch #, bbox_mask # entity_num_batch, entity_mask

    def get_role_logits(self, noun_emb):
        role_logits = self.sr_ae_ol(noun_emb)
        return role_logits

    def calculate_loss_all(self, emb_verb_proj, idx_verb_gt, emb_obj_proj, idx_entity_gt, role_logits, verb_logits, noun_logits,
                           role_gt_type, num_bbox, num_entity_gt):
        '''
        :param verb_batch: [batch,]
        :param roles_batch: [batch, arg_num, role_num(one-hot vec)]
        :param args_batch: [batch, arg_num]
        :param emb_verb_proj: size = [batch_size, ]
        :param emb_verb_gt: size = [batch_size, ]
        :param emb_proj: size = [batch_size, OBJ_NUM, emb_dim]
        :param emb_entity_gt: size = [batch_size, ARG_NUM, emb_dim]
        :param role_logits: size = [batch_size, OBJ_NUM, num_role_types]
        :param role_gt_type: size = [batch_size, ARG_NUM]
        :return:
        '''
        ROLE_NUM = role_logits.size(2)
        BATCH_SIZE = role_logits.size(0)
        OBJ_NUM = role_logits.size(1)
        ARG_NUM = role_gt_type.size(1)

        num_entity_gt = torch.LongTensor(num_entity_gt).to(self.device)

        # loss of verb embedding projection
        cost_verb = torch.mean(- verb_logits[torch.arange(BATCH_SIZE), idx_verb_gt])

        # # Create the mask matrix
        # arg_mask = np.ones(shape=idx_entity_gt.size(), dtype=np.uint8)
        # for i in range(BATCH_SIZE):
        #     # for j in range(role_num_batch[i]):
        #     s_len = int(num_entity_gt[i])
        #     arg_mask[i, 0:s_len] = np.zeros(shape=(s_len), dtype=np.uint8)
        # arg_mask = torch.ByteTensor(arg_mask).to(self.device)
        # bbox_mask = np.ones(shape=emb_obj_proj.size(), dtype=np.uint8)
        # for i in range(BATCH_SIZE):
        #     s_len = int(bbox_num_batch[i])
        #     bbox_mask[i, 0:s_len] = np.zeros(shape=(s_len), dtype=np.uint8)
        # bbox_mask = torch.ByteTensor(bbox_mask).to(self.device)

        # loss of every <bbox-entity> pairs
        # use the noun_logits
        # pairwise_cost_emb = torch.sum(- noun_logits.unsqueeze(2) - emb_entity_gt_masked.unsqueeze(1)).pow(2))  # [batch, obj_num, arg_num]
        pairwise_cost_emb = torch.zeros(BATCH_SIZE, OBJ_NUM, ARG_NUM).to(self.device)
        for i in range(BATCH_SIZE):
            for bbox_idx in range(num_bbox[i]):
                pairwise_cost_emb[i][bbox_idx][:num_entity_gt[i]] = - torch.index_select(noun_logits[i][bbox_idx],
                                                                                         dim=0,
                                                                                         index=idx_entity_gt[i][:num_entity_gt[i]]
                                                                                         )  # [batch, obj_num, arg_num]
        # use the mse loss:
        # emb_entity_proj_masked = emb_obj_proj.masked_fill(bbox_mask.unsqueeze(2).expand_as(emb_obj_proj), 0.) #-np.inf
        # emb_entity_gt_masked = emb_entity_gt.masked_fill(arg_mask.unsqueeze(2).expand_as(emb_entity_gt), 0.) #-np.inf
        # pairwise_cost_emb = torch.sum((emb_entity_proj_masked.unsqueeze(2) - emb_entity_gt_masked.unsqueeze(1)).pow(2), dim=-1) # [batch, obj_num, arg_num]
        # # print('pairwise_cost_emb', pairwise_cost_emb)  # [batch, obj_num, arg_num]

        # loss of every <role-rolegt> pairs (loss of role classifier)
        pairwise_cost_role = torch.zeros(BATCH_SIZE, OBJ_NUM, ARG_NUM).to(self.device)
        for i in range(BATCH_SIZE):
            for bbox_idx in range(num_bbox[i]):
                pairwise_cost_role[i][bbox_idx][:num_entity_gt[i]] = - torch.index_select(role_logits[i][bbox_idx],
                                                                                         dim=0,
                                                                                         index=role_gt_type[i][:num_entity_gt[i]]
                                                                                         )  # [batch, obj_num, arg_num]
        # role_logits_masked = role_logits.masked_fill(bbox_mask.unsqueeze(2).expand_as(role_logits), 0)
        # role_gt_type_masked = role_gt_type.masked_fill(arg_mask, 0)
        # role_gt_norm = role_gt_type_masked.unsqueeze(1).expand(BATCH_SIZE, OBJ_NUM, ARG_NUM)
        # role_logits_softmax = F.log_softmax(role_logits_masked, dim=2)
        # role_logits_softmax_norm = role_logits_softmax.unsqueeze(2).expand(BATCH_SIZE, OBJ_NUM, ARG_NUM, ROLE_NUM)
        # pairwise_cost_role = F.nll_loss(role_logits_softmax_norm.permute(0,3,1,2), role_gt_norm, reduction='none')

        # pairwise_cost = (self.hyperparams['loss_weight_verb'] * pairwise_cost_emb) + (self.hyperparams['loss_weight_noun'] * pairwise_cost_role)
        pairwise_cost = pairwise_cost_emb + pairwise_cost_role
        # print('pairwise_cost', pairwise_cost)  # [batch, obj_num, arg_num]

        pairwise_cost_numpy = pairwise_cost.cpu().detach().numpy()
        cost_entity_emb = 0.0
        cost_role = 0.0
        for i in range(BATCH_SIZE):
            # print('pairwise_cost_numpy[i]', i, pairwise_cost_numpy[i])
            oi, ti = linear_sum_assignment(pairwise_cost_numpy[i])  # use the addition to find the matching pair
            cost_entity_emb = cost_entity_emb + torch.sum(pairwise_cost_emb[i][oi, ti])
            cost_role = cost_role + torch.sum(pairwise_cost_role[i][oi, ti])
        cost_entity_emb = cost_entity_emb / BATCH_SIZE
        cost_role = cost_role / BATCH_SIZE

        loss_terms = {'verb': cost_verb.item(), 'noun': cost_entity_emb.item(), 'role': cost_role}
        loss_sr = (self.hyperparams['loss_weight_verb'] * cost_verb) + (
                    self.hyperparams['loss_weight_noun'] * cost_entity_emb) + (
                    self.hyperparams['loss_weight_noun'] * cost_role)

        return loss_sr, loss_terms, verb_logits, noun_logits, role_logits

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

    def calculate_loss_ace(self, event_logits, event_ae_logits, event_gt_batch, ee_roles_gt_batch, num_bbox, num_entity_gt):
        # calculate_loss_ed(self, logits, mask, label, weight)
        BATCH_SIZE = event_logits.size()[0]
        OBJ_NUM = event_ae_logits.size(1)
        ARG_NUM = ee_roles_gt_batch.size(1)

        # if weight is not None:
        #     weight = weight.to(self.device)
        #     cost_ed = F.nll_loss(F.log_softmax(event_logits, dim=1), event_gt_batch, weight=weight)
        # else:
        cost_ed = F.nll_loss(F.log_softmax(event_logits, dim=1), event_gt_batch)

        num_entity_gt = torch.LongTensor(num_entity_gt).to(self.device)

        # loss of every <role-rolegt> pairs (loss of role classifier)

        pairwise_cost_role = torch.zeros(BATCH_SIZE, OBJ_NUM, ARG_NUM).to(self.device)
        for i in range(BATCH_SIZE):
            for bbox_idx in range(num_bbox[i]):
                pairwise_cost_role[i][bbox_idx][:num_entity_gt[i]] = - torch.index_select(event_ae_logits[i][bbox_idx],
                                                                                          dim=0,
                                                                                          index=ee_roles_gt_batch[i][
                                                                                                :num_entity_gt[i]]
                                                                                          )  # [batch, obj_num, arg_num]
        pairwise_cost_numpy = pairwise_cost_role.cpu().detach().numpy()
        cost_ae = 0.0
        for i in range(BATCH_SIZE):
            # print('pairwise_cost_numpy[i]', i, pairwise_cost_numpy[i])
            oi, ti = linear_sum_assignment(pairwise_cost_numpy[i])  # use the addition to find the matching pair
            cost_ae = cost_ae + torch.sum(pairwise_cost_role[i][oi, ti])
        cost_ae = cost_ae / BATCH_SIZE

        loss_ace_terms = {'ace_ed': cost_ed.item(), 'ace_ae': cost_ae.item()}
        loss_sr_ace = (self.hyperparams['loss_weight_verb'] * cost_ed) + (
                    self.hyperparams['loss_weight_noun'] * cost_ae)

        return loss_sr_ace, loss_ace_terms, event_logits, event_ae_logits

    # def calculate_loss_verb(self, emb_verb_proj, emb_verb_gt):
    #     '''
    #     :param emb_verb_proj: size = [batch_size, emb_dim]
    #     :param emb_verb_gt: size = [batch_size, emb_dim]
    #     :return:
    #     '''
    #     loss = self.proj_loss(emb_verb_proj, emb_verb_gt)
    #     # loss = loss.item()  #loss.detach()
    #     return loss

    # def get_all_verbs_emb(self):
    #     # get all indexes, should include <PAD>
    #     verb_idxes = torch.from_numpy(np.arange(self.hyperparams['wvemb_size'])).to(self.device) # [0,1,....,verb_size-1],
    #     embed_verb_all = self.wvembeddings(verb_idxes)  # batch, verb_size, embed_verb_dim
    #     return embed_verb_all

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
