from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import codecs

import os
import sys
sys.path.append('../../../')
from src.dataflow.numpy.prepare_grounding_JMEE import word2entity

dep_parser = CoreNLPDependencyParser(url='http://localhost:9000', tagtype='ner')
# ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
# sent_tokenizer = PunktSentenceTokenizer()


def generate_json(img_id_list, img_suffix, associated_sentence, sent_idx, sentence_start, sentence_end, example_list):
    '''
    difference between grounding:
    (1) image_list, instead of certain image
    (2) sentence are not parsed using sentence tokenizer
    (3) sent_idx is from the ltf, also sentence start, end
    :param img_id_list:
    :param associated_sentence:
    :param sent_idx:
    :param sentence_start:
    :param sentence_end:
    :param example_list:
    :return:
    '''
    # sentences = sent_tokenizer.span_tokenize(associated_sentence)
    # if sum(1 for _ in sentences) > 1:
    #     # if caption contain multiple sentence, ignore this datapoint
    #     # print('multiple sentences: ', caption)
    #     return example_list
    #
    # sent_idx = 0
    # for sentence_start, sentence_end in sent_tokenizer.span_tokenize(associated_sentence):
        # each sentence
    sen_obj = dict()
    # sen_obj["image"] = img_id
    sen_obj["sentence_id"] = sent_idx  #'%s__%d' % (img_id, sent_idx)
    sen_obj["sentence_start"] = sentence_start
    sen_obj["sentence_end"] = sentence_end
    sen_obj['sentence'] = associated_sentence #associated_sentence[sen_obj["sentence_start"]:sen_obj["sentence_end"]]
    sen_obj['words'] = list()
    sen_obj['index_all'] = list()
    sen_obj['index'] = list()
    sen_obj['pos-tags'] = list()
    sen_obj['stanford-colcc'] = list()
    # sent_idx = sent_idx + 1

    parse, = dep_parser.raw_parse(sen_obj['sentence'])
    pre = 0
    word_idx = 0
    for line in parse.to_conll(4).strip().split('\n'):
        ele = line.split('\t')

        # print(ele)
        word = ele[0].replace('-LRB-', '(').replace('-RRB-', ')')
        sen_obj['words'].append(word)
        bg = sen_obj['sentence'].find(word, pre)
        ed = bg + len(word) - 1
        pre = ed + 1
        # print(len(word), pre)
        # print('\"' + sen_obj['sentence'][bg:ed + 1] + '\"', '\"' + word + '\"')
        # assert sen_obj['sentence'][bg:ed + 1] == word
        if sen_obj['sentence'][bg:ed + 1] != word:
            return example_list
        sen_obj['index_all'].append((bg + sen_obj['sentence_start'], ed + sen_obj['sentence_start']))
        sen_obj['index'].append((bg, ed))

        pos = ele[1]
        sen_obj['pos-tags'].append(pos)

        head = ele[2]
        deprel = ele[3]
        sen_obj['stanford-colcc'].append('%s/dep=%d/gov=%d' % (deprel, word_idx, int(head)-1) )
            # deprel + "/dep=" + str(word_idx) + "/gov=" + str(head-1))
        word_idx = word_idx + 1

    sen_obj['golden-entity-mentions'] = list()
    ners = dep_parser.tag(sen_obj['words'])
    # print('ners', ners)
    # word_num = len(ners)
    # for idx_i in range(word_num):
    #     ner_tag = ners[idx_i][-1]
    #     word = ners[idx_i][0]
    #     if word == sen_obj['words'][idx_i]:
    #         if ner_tag != 'O':
    #             entity_obj = dict()
    #             entity_obj['entity-type'] = ner_tag
    #             entity_obj['text'] = word
    #             entity_obj['start'] = sen_obj['index'][idx_i][0]
    #             entity_obj['end'] = sen_obj['index'][idx_i][1]
    #             sen_obj['golden-entity-mentions'].append(entity_obj)
    #     else:
    #         print('word', word)
    #         print('word_', sen_obj['words'][idx_i])
    ner_info = word2entity(ners, sen_obj['words'], sen_obj['index'])
    # print('ner_info', ner_info)
    for entity_id in ner_info:
        entity_obj = dict()
        entity_obj['entity-type'] = ner_info[entity_id]['type']
        entity_obj['text'] = ' '.join(ner_info[entity_id]['words'])
        entity_obj['start'] = ner_info[entity_id]['start']
        entity_obj['end'] = ner_info[entity_id]['end']
        # entity_obj['start_all'] = ner_info[entity_id]['start'] + sentence_start
        # entity_obj['end_all'] = ner_info[entity_id]['end'] + sentence_start
        sen_obj['golden-entity-mentions'].append(entity_obj)
    # sen_obj['golden-event-mentions'] = list()

    for img_id in img_id_list:
        sen_obj_new = sen_obj.copy()
        sen_obj_new["image"] = img_id+img_suffix
        example_list.append(sen_obj_new)
    # print('sen_obj', sen_obj)
    # sent_idx += 1
    return example_list

def main(image_type_role, doc_ltf_dir, img_suffix, text_json, image_json):
    # get the voa image-sentence pair, each of them contain one sentence from the text, and one image related tp that sentence
    # test is quick , so the duplicated cases are toleratable

    image_set = defaultdict(lambda : defaultdict())
    doc_set = defaultdict(set)
    for line in open(image_type_role):
        line = line.rstrip('\n')
        if not line.startswith('VOA'):
            continue
        tabs = line.split('\t')
        image_id_nosuffix = tabs[0]
        event_type = tabs[1]
        role = tabs[2]
        entity_name = tabs[3]
        xmin = int(tabs[4])
        ymin = int(tabs[5])
        xmax = int(tabs[6])
        ymax = int(tabs[7])

        doc_id = image_id_nosuffix[:image_id_nosuffix.rfind('_')]

        image_set[image_id_nosuffix]['event_type'] = event_type
        if 'role' not in image_set[image_id_nosuffix]:
            image_set[image_id_nosuffix]['role'] = defaultdict(list)
        image_set[image_id_nosuffix]['role'][role].append( (entity_name, xmin, ymin, xmax, ymax) )
        doc_set[doc_id].add(image_id_nosuffix)

    image_json_writer = codecs.open(image_json, 'w', 'utf-8')
    json.dump(image_set, image_json_writer, indent=2)

    # print(doc_set)
    example_list = []
    for doc_id in doc_set:
        print(doc_id)
        tokens = []

        ltf_file_path = os.path.join(doc_ltf_dir, doc_id + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            print('[ERROR]NoLTF %s' % doc_id)
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    seg_beg = int(seg.attrib["start_char"])
                    seg_end = int(seg.attrib["end_char"])
                    seg_id = seg.attrib["id"]
                    for token in seg:
                        if token.tag == "ORIGINAL_TEXT":
                            seg_text = token.text
                            break
                    generate_json(doc_set[doc_id], img_suffix, seg_text, seg_id, seg_beg, seg_end, example_list)

    text_json_writer = codecs.open(text_json, 'w', 'utf-8')
    json.dump(example_list, text_json_writer, indent=2)



if __name__ == "__main__":
    image_type_role = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/anno_image_type_role.txt'
    doc_ltf_dir = '/scratch/manling2/data/mm-event-graph/voa_anno_trimmed/article_ltf'
    img_suffix = '.jpg'
    text_json = '/scratch/manling2/data/mm-event-graph/voa_anno_trimmed/article.json'
    image_json = '/scratch/manling2/data/mm-event-graph/voa_anno_trimmed/image_event.json'


# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 &

    main(image_type_role, doc_ltf_dir, img_suffix, text_json, image_json)


