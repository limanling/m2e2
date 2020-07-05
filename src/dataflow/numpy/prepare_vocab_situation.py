"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter
import csv

import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util import vocab, constant, helper

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('data_dir', default='/dvmm-filer2/users/manling/mm-event-graph/data/imSitu/', help='TACRED directory.')
    # /dvmm-filer2/users/manling/mm-event-graph/data/imSitu/
    parser.add_argument('object_data_dir', default='/dvmm-filer2/users/manling/mm-event-graph/data/object/', help='Object Detection directory.')
    # /dvmm-filer2/users/manling/mm-event-graph/data/object/
    parser.add_argument('vocab_dir', default='/dvmm-filer2/users/manling/mm-event-graph/data/vocab', help='Output vocab directory.')
    # /dvmm-filer2/users/manling/mm-event-graph/data/vocab
    parser.add_argument('--glove_dir', default='/dvmm-filer2/users/manling/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    imsitu_ontology_file = args.data_dir + '/imsitu_space.json'
    object_label_file = args.object_data_dir +'/class-descriptions-boxable.csv'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file_noun = args.vocab_dir + '/vocab_situation_noun.pkl'
    emb_file_noun = args.vocab_dir + '/embedding_situation_noun.npy'
    vocab_file_role = args.vocab_dir + '/vocab_situation_role.pkl'
    emb_file_role = args.vocab_dir + '/embedding_situation_role.npy'
    vocab_file_verb = args.vocab_dir + '/vocab_situation_verb.pkl'
    emb_file_verb = args.vocab_dir + '/embedding_situation_verb.npy'

    # load files
    print("loading files...")
    imsitu_info = json.load(open(imsitu_ontology_file))
    train_tokens_noun, train_tokens_verb, role_tokens = load_tokens_image_situation(train_file, imsitu_info)
    # dev_tokens_noun, dev_tokens_verb, role_tokens = load_tokens_image_situation(dev_file, imsitu_info)
    # test_tokens_noun, test_tokens_verb, role_tokens = load_tokens_image_situation(test_file, imsitu_info)
    print(len(train_tokens_noun), len(train_tokens_verb), len(role_tokens))
    # add object labels
    train_tokens_noun.extend(load_object_detection_labels(object_label_file))
    if args.lower:
        # train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
        #         (train_tokens, dev_tokens, test_tokens)]
        train_tokens_noun = [[t.lower() for t in tokens] for tokens in train_tokens_noun]
    print(len(train_tokens_noun), len(train_tokens_verb), len(role_tokens))
    print(len(set(train_tokens_noun)), len(set(train_tokens_verb)), len(set(role_tokens)))

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))

    print("building vocab...")
    v_noun = build_vocab(train_tokens_noun, glove_vocab, 0, add_entity_mask=False)
    v_verb = build_vocab(train_tokens_verb, glove_vocab, 0, add_entity_mask=False)

    print("building role vocab...")
    v_role = build_vocab(role_tokens, glove_vocab, 0, add_entity_mask=False)

    print("calculating oov...")
    datasets = {'train': train_tokens_noun}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v_noun)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building noun embeddings...")
    embedding_noun = vocab.build_embedding(wv_file, v_noun, wv_dim)
    print("embedding size: {} x {}".format(*embedding_noun.shape))

    print("building verb embeddings...")
    embedding_verb = vocab.build_embedding(wv_file, v_verb, wv_dim)
    print("embedding size: {} x {}".format(*embedding_verb.shape))

    print("building role embeddings...")
    embedding_role = vocab.build_embedding(wv_file, v_role, wv_dim)
    print("embedding size: {} x {}".format(*embedding_role.shape))

    print("dumping to files...")
    with open(vocab_file_noun, 'wb') as outfile:
        pickle.dump(v_noun, outfile)
    np.save(emb_file_noun, embedding_noun)

    print("dumping to files...")
    with open(vocab_file_verb, 'wb') as outfile:
        pickle.dump(v_verb, outfile)
    np.save(emb_file_verb, embedding_verb)

    print("dumping to files...")
    with open(vocab_file_role, 'wb') as outfile:
        pickle.dump(v_role, outfile)
    np.save(emb_file_role, embedding_role)

    print("all done.")

# def load_tokens(filename):
#     with open(filename) as infile:
#         data = json.load(infile)
#         tokens = []
#         for d in data:
#             ts = d['token']
#             ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
#             # do not create vocab for entity words
#             ts[ss:se+1] = ['<PAD>']*(se-ss+1)
#             ts[os:oe+1] = ['<PAD>']*(oe-os+1)
#             tokens += list(filter(lambda t: t!='<PAD>', ts))
#     print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
#     return tokens
def load_tokens_image_situation(imsitu_annotation_file, imsitu_info):
    tokens_noun = []
    tokens_verb = []
    ts_noun = []
    ts_verb = []
    roles = []
    nouns = imsitu_info["nouns"]
    for noun in nouns:
        for noun_str in nouns[noun]['gloss']:
            # for noun_word in noun_str.split(' '):
            #     ts.append(noun_word)
            ts_noun.append(noun_str)
    verbs = imsitu_info["verbs"]
    for verb in verbs:
        ts_verb.append(verb)
        roles.extend(verbs[verb]['order'])
    tokens_noun = list(filter(lambda t: t != '<PAD>', ts_noun))
    tokens_verb = list(filter(lambda t: t != '<PAD>', ts_verb))
    # parsed_data = json.load(open(imsitu_annotation_file))
    # ts = []
    # for image_id in parsed_data:
    #     verb = parsed_data[image_id]['verb']
    #     ts.append(verb)
    #     # ontology
    #     ts.extend(verbs[verb]['order'])
    #     # role values
    #     frames = parsed_data[image_id]['frames']
    #     for frame in frames:
    #         for role in frame:
    #             role_value_id = frame[role]
    #             if len(role_value_id) > 0:
    #                 role_value = nouns[role_value_id]['gloss']
    #                 ts.append(role_value)
    #     tokens += list(filter(lambda t: t != '<PAD>', ts))
    # print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(parsed_data), filename))
    return tokens_noun, tokens_verb, roles

def load_object_detection_labels(class_map_file):
    label_name = []
    with open(class_map_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            label_name.append(row[1])
    return label_name

def build_vocab(tokens, glove_vocab, min_freq, add_entity_mask=False):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        # v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
        v = sorted([t for t in counter], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    if add_entity_mask:
        v = constant.VOCAB_PREFIX + entity_masks() + v
    else:
        v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    # print(c)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """ Get all entity mask tokens as a list. Accoeding to constant.py"""
    masks = []
    # subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    # masks += ["SUBJ-" + e for e in subj_entities]
    # masks += ["OBJ-" + e for e in obj_entities]
    masks += [e.lower() for e in obj_entities]
    print(masks) #['SUBJ-ORGANIZATION', 'SUBJ-PERSON', 'OBJ-PERSON', 'OBJ-ORGANIZATION', 'OBJ-DATE', 'OBJ-NUMBER', 'OBJ-TITLE', 'OBJ-COUNTRY', 'OBJ-LOCATION', 'OBJ-CITY', 'OBJ-MISC', 'OBJ-STATE_OR_PROVINCE', 'OBJ-DURATION', 'OBJ-NATIONALITY', 'OBJ-CAUSE_OF_DEATH', 'OBJ-CRIMINAL_CHARGE', 'OBJ-RELIGION', 'OBJ-URL', 'OBJ-IDEOLOGY']
    return masks

if __name__ == '__main__':
    main()


# 82153 tokens from 3334 examples loaded from /data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/ace/JMEE_data/head/JMEE_train.json.
# 9751 tokens from 347 examples loaded from /data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/ace/JMEE_data/head/JMEE_dev.json.
# 8728 tokens from 293 examples loaded from /data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/ace/JMEE_data/head/JMEE_test.json.
# loading glove...
# 2195893 words loaded from glove.
# building vocab...
# ['Weapon', 'Organization', 'Geopolitical_Entity', 'Location', 'Time', 'Vehicle', 'Value', 'Facility', 'Person']
# vocab built with 9396/9626 words.
# calculating oov...
# train oov: 288/82153 (0.35%)
# dev oov: 844/9751 (8.66%)
# test oov: 992/8728 (11.37%)
# building embeddings...
# embedding size: 9396 x 300


