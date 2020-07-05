"""
Data loader for ACE json files.
Need to construct the negative samples
"""
import json
import random
import numpy as np
import codecs

from src.util import constant, helper, vocab
from collections import defaultdict

class ACERoleDataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    word_dropout default=0.04, help='The rate at which randomly set a word to UNK.'
    """
    def __init__(self, filename, batch_size, vocab, LABEL_TO_ID_TRIGGER, LABEL_TO_ID_ROLE,
                 sent_vectorize=False, lower=True, shuffle=False, evaluation=False, word_dropout=0.04):
        self.batch_size = batch_size
        # self.opt = opt
        self.vocab = vocab
        self.sent_vectorize = sent_vectorize
        self.eval = evaluation
        self.word_dropout = word_dropout
        self.label2id_trigger = LABEL_TO_ID_TRIGGER
        self.label2id_role = LABEL_TO_ID_ROLE
        self.ner_norm = constant.NER_NORM

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.parse_example(data, vocab, lower)
        # print('data[0]', data[0])

        # shuffle for training
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label_trigger = dict([(v,k) for k,v in self.label2id_trigger.items()])
        # self.labels_trigger = [self.id2label_trigger[d[-2]] for d in data]
        self.id2label_role = dict([(v, k) for k, v in self.label2id_role.items()])
        # # self.labels_role = [self.id2label_role[d[-1]] for d in data]
        # self.labels_role = []
        # for d in data:
        #     self.labels_role.append([])
        #     for (role, obj_positions, obj_ner) in d[-1]:
        #         self.labels_role[-1].append(self.id2label_role[role])
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def parse_example(self, path, amr):
        examples = []

        _file = codecs.open(path, 'r', 'utf-8')
        jl = json.load(_file)
        print(path, len(jl))
        for js in jl:
            ex = self.parse_sentence(js, fields, amr)
            if ex is not None:
                examples.append(ex)
            # for line in f:
            #     line = line.strip()
            #     if len(line) == 0:
            #         continue
            #     print(line)
            #     jl = json.loads(line, encoding="utf-8")
            #     for js in jl:
            #         ex = self.parse_sentence(js, fields)
            #         if ex is not None:
            #             examples.append(ex)

        return examples

    def parse_sentence(self, js, fields, amr):
        WORDS = fields["words"]
        POSTAGS = fields["pos-tags"]
        # LEMMAS = fields["lemma"]
        ENTITYLABELS = fields["golden-entity-mentions"]
        if amr:
            colcc = "amr-colcc"
        else:
            colcc = "stanford-colcc"
        # print(colcc)
        ADJMATRIX = fields[colcc]
        LABELS = fields["golden-event-mentions"]
        EVENTS = fields["all-events"]
        ENTITIES = fields["all-entities"]

        sentence = Sentence_ace(json_content=js, graph_field_name=colcc)
        ex = Example()
        word_preprocess(sentence.wordList) #setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.wordList))
        setattr(ex, POSTAGS[0], POSTAGS[1].preprocess(sentence.posLabelList))
        # setattr(ex, LEMMAS[0], LEMMAS[1].preprocess(sentence.lemmaList))
        setattr(ex, ENTITYLABELS[0], ENTITYLABELS[1].preprocess(sentence.entityLabelList))
        setattr(ex, ADJMATRIX[0], (sentence.adjpos, sentence.adjv))
        setattr(ex, LABELS[0], LABELS[1].preprocess(sentence.triggerLabelList))
        setattr(ex, EVENTS[0], EVENTS[1].preprocess(sentence.events))
        setattr(ex, ENTITIES[0], ENTITIES[1].preprocess(sentence.entities))

        if self.keep_events is not None:
            if self.only_keep and sentence.containsEvents != self.keep_events:
                return None
            elif not self.only_keep and sentence.containsEvents < self.keep_events:
                return None
            else:
                return ex
        else:
            return ex

    def longest(self):
        return max([len(x.POSTAGS) for x in self.examples])

    def word_preprocess(self, x):
        # if self.sequential and isinstance(x, six.text_type):
        #     x = self.tokenize(x.rstrip('\n'))
        # if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    # def preprocess(self, parsed_data, vocab, lower=True, anonymize=True, include_other=False):
    #     """ Preprocess the data and convert to ids. """
    #
    #     processed = []
    #     for sent in parsed_data:
    #         tokens = list(sent['words'])
    #         if lower:
    #             tokens = [t.lower() for t in tokens]
    #         l = len(tokens)
    #         # get entity type
    #         entities = sent['golden-entity-mentions']
    #         entity_dict = defaultdict(lambda : defaultdict(list))
    #         ner_list = [constant.PAD_TOKEN] * l
    #         for entity in entities:
    #             entity_start = entity['start']
    #             entity_end = entity['end']
    #             entity_type = self.ner_norm[entity['entity-type'].split(':')[0]]
    #             entity_dict[entity_start][entity_end] = entity_type
    #             for i in range(entity_start, entity_end):
    #                 ner_list[i] = entity_type
    #         ner = map_to_ids(ner_list, constant.NER_TO_ID)
    #         ## get NLP features
    #         # pos = map_to_ids(sent['pos-tags'], constant.POS_TO_ID)
    #         # dep_rel_dict = [constant.PAD_TOKEN] * l
    #         head = [0] * l
    #         for dep_info in sent['stanford-colcc']:
    #             # print(dep_info)
    #             tabs = dep_info.split('/')
    #             idx = int(tabs[1].replace("dep=", ""))
    #             head_idx = int(tabs[2].replace("gov=", "")) + 1
    #             head[idx] = head_idx
    #             # dep_rel = tabs[0]
    #             # dep_rel_dict[idx] = dep_rel
    #         # deprel = map_to_ids(dep_rel_dict, constant.DEPREL_TO_ID)
    #         assert any([x == 0 for x in head])
    #
    #
    #         events = sent['golden-event-mentions']
    #         for event in events:
    #             # print('--------------')
    #             # args_idx_list = list()
    #             role_list = list()
    #             role_neg_list = list()
    #             obj_position_list = list()
    #             object_type_list = list()
    #
    #             event_type = event['event_type']
    #             if event_type in constant.LABEL_TO_ID_TRIGGER_UNVISUAL:
    #                 continue
    #             elif event_type in constant.LABEL_TO_ID_TRIGGER_UNSEEN:
    #                 if include_other:
    #                     event_type = constant.NEGATIVE_LABEL_TRIGGER
    #                 else:
    #                     continue
    #             event_idx = self.label2id_trigger[event_type]
    #             # construct event_neg
    #             event_neg = random.choice(list(self.label2id_trigger.keys()))
    #             while event_neg == event_type \
    #                     or event_neg == constant.PAD_TOKEN \
    #                     or event_neg == constant.NEGATIVE_LABEL_TRIGGER:
    #                 event_neg = random.choice(list(self.label2id_trigger.keys()))
    #             event_neg_idx = self.label2id_trigger[event_neg]
    #
    #             if anonymize:
    #                 tokens_event = tokens.copy()
    #             else:
    #                 tokens_event = tokens
    #
    #             trigger_start = event['trigger']['start']
    #             trigger_end = event['trigger']['end']
    #             ss, se = trigger_start, (trigger_end-1) # inclusive offset
    #             subj_positions = get_positions(ss, se, l)
    #             # arg_dict = defaultdict(lambda: defaultdict(list))
    #             for arg in event['arguments']:
    #                 arg_start = arg['start']
    #                 arg_end = arg['end']
    #                 # role = '%s_%s' % (event_type, arg['role'])
    #                 role = arg['role']
    #                 if role in constant.LABEL_TO_ID_ROLE_UNVISUAL:
    #                     # print(role)
    #                     continue
    #                 elif role in constant.LABEL_TO_ID_ROLE_UNSEEN:
    #                     if include_other:
    #                         role = constant.NEGATIVE_LABEL_ROLE
    #                     else:
    #                         continue
    #                 role_idx = self.label2id_role[role]
    #                 # arg_dict[arg_start][arg_end] = role
    #                 # construct role_neg
    #                 role_neg = random.choice(list(self.label2id_role.keys()))  # !!sample from event type related roles
    #                 while role_neg == role \
    #                     or role_neg == constant.PAD_TOKEN \
    #                     or role_neg == constant.NEGATIVE_LABEL_ROLE:
    #                     role_neg = random.choice(list(self.label2id_role.keys()))
    #                 role_neg_idx = self.label2id_role[role_neg]
    #
    #                 os, oe = arg_start, (arg_end - 1)
    #                 obj_positions = get_positions(os, oe, l)
    #                 arg_type = entity_dict[arg_start][arg_end]
    #                 obj_ner_idx = constant.OBJ_NER_TO_ID[arg_type]
    #                 if anonymize:
    #                     tokens_event[os:oe + 1] = [arg_type] * (oe - os + 1)
    #                 if self.sent_vectorize:
    #                     tokens_event = map_to_ids(tokens_event, vocab.word2id)
    #                 role_list.append(role_idx)
    #                 role_neg_list.append(role_neg_idx)
    #                 obj_position_list.append(obj_positions)
    #                 object_type_list.append(obj_ner_idx)
    #
    #             if len(role_list) > 0:
    #                 processed += [(tokens_event, ner, head, subj_positions,
    #                                obj_position_list, object_type_list,
    #                                event_idx, event_neg_idx, role_list, role_neg_list)]
    #                 del tokens_event
    #     # print('processed[0]', processed[0])
    #     return processed
    #



    # def gold(self):
    #     """ Return gold labels as a list. """
    #     return self.labels_trigger, self.labels_role

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        # print('self.data[0]', self.data[0])
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10
        # print('batch', batch)

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            if self.sent_vectorize:
                words_batch = [word_dropout(sent, self.word_dropout, constant.UNK_ID) for sent in batch[0]]
            else:
                words_batch = [word_dropout(sent, self.word_dropout, constant.UNK_TOKEN) for sent in batch[0]]
        else:
            words_batch = batch[0]

        # convert to tensors
        max_sent_len = max(len(x) for x in words_batch)

        if self.sent_vectorize:
            words_batch = get_np(words_batch, batch_size, max_sent_len)  # batch, sent_len
        # pos_batch = get_long_tensor(batch[1], batch_size, max_sent_len)
        ner_batch = get_np(batch[1], batch_size, max_sent_len)   # batch, sent_len
        # deprel_batch = get_long_tensor(batch[3], batch_size, max_sent_len)
        mask_ner_batch_bool = np.equal(ner_batch, np.zeros(ner_batch.shape))  # batch, sent_len
        mask_ner_batch = mask_ner_batch_bool.astype(int)
        head_batch = get_np(batch[2], batch_size, max_sent_len)  # batch, sent_len
        subj_position_batch = get_np(batch[3], batch_size, max_sent_len)    # batch, sent_len
        event_batch = np.array(batch[6])
        event_neg_batch = np.array(batch[7])

        roles_batch = batch[8]  # batch, role_num
        # print('roles_batch', roles_batch)
        max_role_num = max(len(roles) for roles in roles_batch)
        # print('max_role_num', max_role_num)
        roles_batch = get_np(roles_batch, batch_size, max_role_num)
        roles_neg_batch = get_np(batch[9], batch_size, max_role_num)  # batch, role_num
        # print('role finished')

        obj_positions_batch = batch[4]  # batch, role_num, sent_len
        obj_positions_vecs = [get_np(obj_positions, max_role_num, max_sent_len) for obj_positions in obj_positions_batch]
        obj_positions_vec_batch = np.array(obj_positions_vecs)
        obj_ners_batch = get_np(batch[5], batch_size, max_role_num)
        # print('obj finished')

        return (words_batch, mask_ner_batch, ner_batch, head_batch, subj_position_batch,
                obj_positions_vec_batch, obj_ners_batch,
                event_batch, event_neg_batch, roles_batch, roles_neg_batch, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_np(feature_list, batch_size, max_feature_len):
    """ Convert list of list of features to a padded LongTensor. """
    features = np.full((batch_size, max_feature_len), constant.PAD_ID)
    # print('feature_list', feature_list)
    for i, s in enumerate(feature_list):
        features[i, :len(s)] = s
    return features

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout, replacement):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    # return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
    #         else x for x in tokens]
    return [replacement if x != replacement and np.random.random() < dropout \
                else x for x in tokens]

def unpack_batch(batch):
    # inputs = [b.to(device) for b in batch[:9]]
    # labels = batch[9:].to(device)
    words_batch = batch[0]  # not tensor, a list, raw sents

    mask_batch, ner_batch, head_batch, subj_position_batch, obj_positions_batch, obj_ners_batch, \
        event_batch, event_neg_batch, roles_batch, roles_neg_batch \
        = [b for b in batch[1:11]]
    orig_idx = batch[11]
    
    # lens = head_batch.eq(0).long().sum(1).squeeze()  # batch, 1
    # return inputs, labels, tokens, head, subj_position, obj_position, lens
    return words_batch, mask_batch, ner_batch, head_batch, subj_position_batch, \
           obj_positions_batch, obj_ners_batch, \
           event_batch, event_neg_batch, roles_batch, roles_neg_batch, orig_idx