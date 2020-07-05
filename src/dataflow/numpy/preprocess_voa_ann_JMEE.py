from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser
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
from src.dataflow.numpy.anno_mapping import entity_type_mapping_brat, event_type_mapping_brat2ace, event_role_mapping_brat2ace

dep_parser = CoreNLPDependencyParser(url='http://localhost:9000') #, tagtype='ner')
# ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
# sent_tokenizer = PunktSentenceTokenizer()
# parser = CoreNLPParser('http://localhost:9000')


def mapping_type(line):
    line = line.replace(
        'Conflict_Attack\tattacker',
        'Conflict||Attack\tAttacker'.upper()
    ).replace(
        'Conflict_Attack\tinstrument',
        'Conflict||Attack\tInstrument'.upper()
    ).replace(
        'Conflict_Attack\tparticipant',
        'Conflict||Attack\tAttacker'.upper()
    ).replace(
        'Conflict_Attack\tplace',
        'Conflict||Attack\tPlace'.upper()
    ).replace(
        'Conflict_Attack\ttarget',
        'Conflict||Attack\tTarget'.upper()
    ).replace(
        'Conflict_Attack\tvictim',
        'Conflict||Attack\tTarget'.upper()
    ).replace(
        'Conflict_Demonstrate\tdemonstrator',
        'Conflict||Demonstrate\tEntity'.upper()
    ).replace(
        'Conflict_Demonstrate\tplace',
        'Conflict||Demonstrate\tplace'.upper()
    ).replace(
        'Contact_Meet\tparticipant',
        'Contact||Meet\tEntity'.upper()
    ).replace(
        'Justice_ArrestJail\tagent',
        'Justice||Arrest|Jail\tAgent'.upper()
    ).replace(
        'Justice_ArrestJail\tperson',
        'Justice||Arrest|Jail\tAgent'.upper()
    ).replace(
        'Life_Die\tplace',
        'Life||Die\tplace'.upper()
    ).replace(
        'Life_Die\tvictim',
        'Life||Die\tVictim'.upper()
    ).replace(
        'Movement_TransportPerson\tinstrument',
        'Movement||Transport\tVehicle'.upper()
    ).replace(
        'Movement_TransportPerson\tperson',
        'Movement||Transport\tArtifact'.upper()
    )
    return line


def generate_json(img_id_list, img_suffix, sent_text, associated_sentence, sent_idx, sentence_start, sentence_end,
                  ner_info, ner_sent_index, event_info, event_sent_index, example_list, ltf_tokens=None):
    '''
    difference between grounding:
    (1) image_list, instead of certain image
    (2) sentence are not parsed using sentence tokenizer
    (3) sent_idx is from the ltf, also sentence start, end
    :param img_id_list:
    :param associated_sentence: list
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
    sen_obj['sentence'] = sent_text #' '.join(associated_sentence) #associated_sentence[sen_obj["sentence_start"]:sen_obj["sentence_end"]]
    sen_obj['words'] = list()
    sen_obj['index_all'] = list()
    sen_obj['index'] = list()
    sen_obj['pos-tags'] = list()
    sen_obj['stanford-colcc'] = list()
    # sent_idx = sent_idx + 1

    # result_dict = parser.api_call(sent_text)

    pre = 0
    parse, = dep_parser.parse(associated_sentence)
    word_idx = 0
    parsed_tokens = parse.to_conll(4).strip().split('\n')
    for line in parsed_tokens:
        ele = line.split('\t')
    # for parsed_result in dep_parser.parse(associated_sentence):
    #     for address in parsed_result:
    #         word_idx = address - 1
        # if ltf_tokens is None:
        word = ele[0].replace('-LRB-', '(').replace('-RRB-', ')')
        # word = parsed_result[address]['word']
        sen_obj['words'].append(word)
        bg = sen_obj['sentence'].find(word, pre)
        ed = bg + len(word) - 1
        pre = ed #+ 1 # no `+1`
        if sen_obj['sentence'][bg:ed + 1] != word:
            print('[ERROR] offset not consistent, ', word, bg, ed, sent_text, associated_sentence)
            return example_list
        sen_obj['index_all'].append((bg + sen_obj['sentence_start'], ed + sen_obj['sentence_start']))
        sen_obj['index'].append((bg, ed))
        # print(sen_obj['words'])
        # print(sen_obj['index_all'])
        # else:
        #     if len(ele) != len(ltf_tokens):
        #         print('[ERROR] different tokenization')
        #         print(parsed_tokens)
        #         print(ltf_tokens)
        #     word, bg_all, ed_all = ltf_tokens[word_idx]
        #     sen_obj['words'].append(word)
        #     sen_obj['index_all'].append( (bg_all, ed_all) )
        #     sen_obj['index'].append((bg_all - sen_obj['sentence_start'], ed_all - sen_obj['sentence_start']))


        pos = ele[1] #parsed_result[address]['tag'] #
        sen_obj['pos-tags'].append(pos)

        head = ele[2] #parsed_result[address]['head'] #
        deprel = ele[3] #parsed_result[address]['rel'] #
        sen_obj['stanford-colcc'].append('%s/dep=%d/gov=%d' % (deprel, word_idx, int(head)-1) )
        word_idx = word_idx + 1



    sen_obj['golden-entity-mentions'] = list()
    # ners = dep_parser.tag(sen_obj['words'])
    # ner_info = word2entity(ners, sen_obj['words'], sen_obj['index'])
    # print('ner_info', ner_info)
    for entity_id in ner_sent_index:
        entity_obj = dict()
        entity_obj['entity-type'] = ner_info[entity_id]['type']
        entity_obj['text'] = ner_info[entity_id]['words']
        entity_obj['start'] = ner_sent_index[entity_id][0]
        entity_obj['end'] = ner_sent_index[entity_id][-1] + 1
        # entity_obj['start_all'] = ner_info[entity_id]['start'] + sentence_start
        # entity_obj['end_all'] = ner_info[entity_id]['end'] + sentence_start
        sen_obj['golden-entity-mentions'].append(entity_obj)
    # sen_obj['golden-event-mentions'] = list()

    sen_obj['golden-event-mentions'] = []
    for event_id in event_sent_index:
        event_obj = dict()
        event_obj['trigger'] = dict()
        event_obj['trigger']['start'] = event_sent_index[event_id][0]
        event_obj['trigger']['end'] = event_sent_index[event_id][-1] + 1
        event_obj['trigger']['text'] = event_info[event_id]['trigger']

        event_obj['event_type'] = event_info[event_id]['type']

        event_obj['arguments'] = list()

        if 'arguments' in event_info[event_id]:
            for role_name in event_info[event_id]['arguments']:
                role_entity_id = event_info[event_id]['arguments'][role_name]
                if role_entity_id not in ner_sent_index:
                    continue
                role_dict = dict()
                role_dict['role'] = role_name
                role_dict['start'] = ner_sent_index[role_entity_id][0]
                role_dict['end'] = ner_sent_index[role_entity_id][-1] + 1
                role_dict['text'] = ner_info[entity_id]['words']
                event_obj['arguments'].append(role_dict)
                # print(event_info[event_id]['arguments'][role_name], ner_info[role_entity_id])
        else:
            print(event_id, event_info[event_id])
        sen_obj['golden-event-mentions'].append(event_obj)

    # for img_id in img_id_list:
    #     sen_obj_new = sen_obj.copy()
    #     sen_obj_new["image"] = img_id+img_suffix
    #     example_list.append(sen_obj_new)
    sen_obj["image"] = img_id_list  # ""   #img_id+img_suffix
    example_list.append(sen_obj)
    return example_list


def get_ner_event_from_ann(doc_ann_dir, doc_id):
    ann_file_path = os.path.join(doc_ann_dir, doc_id + '.rsd.ann')
    if not os.path.exists(ann_file_path):
        print('[ERROR]NoANN %s' % doc_id)
        return None, None
    ner_info = defaultdict(lambda: defaultdict())  # entity_id -> entities
    event_info = defaultdict(lambda: defaultdict())  # entity_id -> entities
    for line in open(ann_file_path):
        line = line.rstrip('\n')
        if line.startswith('T'):
            # mention, event/entity
            tabs = line.split('\t')
            id = tabs[0]
            subs = tabs[1].split(' ')
            type = subs[0]
            start = int(subs[1])
            end = int(subs[2])
            mention = tabs[2]
            if type in entity_type_mapping_brat:
                ner_info[id]['type'] = entity_type_mapping_brat[type]
                ner_info[id]['start'] = start
                ner_info[id]['end'] = end
                ner_info[id]['words'] = mention
                # ner_info[id]['index'] = []
            elif type in event_type_mapping_brat2ace:
                event_info[id]['type'] = event_type_mapping_brat2ace[type]
                event_info[id]['start'] = start
                event_info[id]['end'] = end
                event_info[id]['trigger'] = mention
            else:
                print('ignored type:', line)
        elif line.startswith('E'):
            # T44	Attack 6595 6604	onslaught
            # E11	Attack:T44 Attacker:T26493 Target:T45
            # print(line)
            role_values = line.split('\t')[-1].split(' ')
            event_id = role_values[0].split(':')[1]
            event_type = role_values[0].split(':')[0]
            if event_id not in event_info:
                continue
            if event_type not in event_role_mapping_brat2ace:
                print('ignored event: ', event_type, event_id)
                continue
            if 'arguments' not in event_info[event_id]:
                event_info[event_id]['arguments'] = dict()
            for role_value in role_values[1:]:
                role_name_raw = role_value.split(':')[0].replace('2', '').replace('3', '').replace('4', '').replace('5', '')
                if len(role_name_raw) == 0:
                    break
                if role_name_raw not in event_role_mapping_brat2ace[event_type]:
                    print('ignored role: ', event_id, role_value)
                    continue
                role_name = event_role_mapping_brat2ace[event_type][role_name_raw]
                entity = role_value.split(':')[1]
                event_info[event_id]['arguments'][role_name] = entity

    return ner_info, event_info

def load_text(image_anno_dir, doc_image_mapping_file, image_dir):
    doc_image_mapping = json.load(open(doc_image_mapping_file))

    doc_set = defaultdict(set)
    for image_xml in os.listdir(image_anno_dir):
        if not image_xml.endswith('.xml'):
            continue
        image_id_nosuffix = image_xml.replace('.xml', '')
        doc_id = image_id_nosuffix[:image_id_nosuffix.rfind('_')]

        if doc_id in doc_set:
            continue

        # get all images of doc_id from metadata
        for image_id in doc_image_mapping[doc_id]:
            # if len(doc_image_mapping[doc_id][image_id]['cap']) > 0:
            image_suffix = doc_image_mapping[doc_id][image_id]['url']
            image_suffix = image_suffix[image_suffix.rfind('.'):]
            if os.path.exists(os.path.join(image_dir, doc_id+'_'+image_id+image_suffix)):
                doc_set[doc_id].add(doc_id+'_'+image_id+image_suffix)
            else:
                print('image not exist', os.path.join(image_dir, doc_id+'_'+image_id+image_suffix))

    return doc_set

def load_image_sent_pair(image_type_role, doc_ltf_dir, doc_ann_dir, img_suffix, text_json, image_json):
    '''
    handle the excel version annotation result

    :param image_type_role:
    :param doc_ltf_dir:
    :param doc_ann_dir:
    :param img_suffix:
    :param text_json:
    :param image_json:
    :return:
    '''
    # get the voa image-sentence pair, each of them contain one sentence from the text, and one image related tp that sentence
    # test is quick , so the duplicated cases are toleratable

    image_set = defaultdict(lambda : defaultdict())
    doc_set = defaultdict(set)
    for line in open(image_type_role):
        line = line.rstrip('\n')

        line = mapping_type(line)

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
        doc_set[doc_id].add(image_id_nosuffix+img_suffix)

    image_json_writer = codecs.open(image_json, 'w', 'utf-8')
    json.dump(image_set, image_json_writer, indent=2)

    return doc_set

def main(doc_set, doc_ltf_dir, doc_ann_dir, img_suffix, text_json):

    # get the voa image-sentence pair, each of them contain one sentence from the text, and one image related tp that sentence
    # test is quick , so the duplicated cases are toleratable

    # print(doc_set)
    example_list = []
    for doc_id in doc_set:
        print(doc_id)
        #

        ner_info, event_info = get_ner_event_from_ann(doc_ann_dir, doc_id)
        if ner_info is None:
            continue

        ltf_file_path = os.path.join(doc_ltf_dir, doc_id + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            print('[ERROR]NoLTF %s' % ltf_file_path)
            continue
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    seg_beg = int(seg.attrib["start_char"])
                    seg_end = int(seg.attrib["end_char"])
                    seg_id = seg.attrib["id"]
                    ner_sent_index = defaultdict(list)
                    event_sent_index = defaultdict(list)
                    tokens = []
                    ltf_tokens = []
                    for token in seg:
                        if token.tag == "ORIGINAL_TEXT":
                            sent_text = token.text
                            # break
                        else:
                            token_beg = int(token.attrib["start_char"])
                            token_end = int(token.attrib["end_char"])
                            token_id = int(token.attrib["id"].split('-')[-1])
                            token_word = token.text
                            tokens.append(token_word)
                            ltf_tokens.append( (token_word, token_beg, token_end) )
                            for entity_id in ner_info:
                                if ner_info[entity_id]['start'] <= token_beg and ner_info[entity_id]['end'] - 1 >= token_end:
                                    ner_sent_index[entity_id].append(token_id)
                            if event_info is not None:
                                for event_id in event_info:
                                    if event_info[event_id]['start'] <= token_beg and event_info[event_id]['end'] - 1 >= token_end:
                                        event_sent_index[event_id].append(token_id)

                    generate_json(list(doc_set[doc_id]), img_suffix, sent_text, tokens, seg_id, seg_beg, seg_end, ner_info, ner_sent_index,
                                  event_info, event_sent_index, example_list, ltf_tokens=ltf_tokens)

        # break
    text_json_writer = codecs.open(text_json, 'w', 'utf-8')
    json.dump(example_list, text_json_writer, indent=2)


if __name__ == "__main__":
    # image_type_role = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/anno_image_type_role.txt'
    # doc_ltf_dir = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/text_all/article_ltf'
    doc_ltf_dir = '/scratch/manling2/data/mm-event-graph/voa/ltf'
    doc_ann_dir = '/scratch/manling2/data/brat-v1.3_Crunchy_Frog/data/batch_all/anno_combine'
    # doc_ann_dir = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/text_all/article_ann' #VOA_EN_NW_2016.11.29.3616534
    doc_image_mapping_file = '/scratch/manling2/data/mm-event-graph/voa/rawdata/voa_img_dataset.json'
    image_anno_dir = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/image_all_final/batch_all/image_entity'
    image_dir = '/scratch/manling2/data/mm-event-graph/voa/rawdata/VOA_image_en/'

    img_suffix = '.jpg'
    text_json = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/article_0705-no1.json'
    doc_list_file = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/doc_list.txt'

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 &

    # doc_set = load_image_sent_pair(image_type_role, doc_ltf_dir, doc_ann_dir, img_suffix, text_json, image_json)
    doc_set = load_text(image_anno_dir, doc_image_mapping_file, image_dir)
    main(doc_set, doc_ltf_dir, doc_ann_dir, img_suffix, text_json)

    # save doc_id list
    writer = open(doc_list_file, 'w')
    # for docid in doc_set:
    writer.write('\n'.join(doc_set))
    writer.flush()
    writer.close()


