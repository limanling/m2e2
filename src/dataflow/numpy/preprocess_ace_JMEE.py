# -*- coding: utf-8 -*-  
import json
import codecs
from stanfordcorenlp import StanfordCoreNLP
import xml.etree.ElementTree as ET
import re
import sys
import ipdb


def load_split_file(method='traditional'):
    if method == 'traditional':
        train_file_path = 'ACE_training.txt'
        dev_file_path = 'ACE_dev.txt'
        test_file_path = 'ACE_test.txt'
        path_list = [train_file_path, dev_file_path, test_file_path]
        file_list = [[] for i in range(3)]
        for i in range(3):
            _file = codecs.open(path_list[i], 'r', 'utf-8')
            for line in _file.readlines():
                file_list[i].append(line.strip())
            _file.close()
        return file_list
    else:
        print('not implement yet: 5-fold')


def generate_example_list(file_path, id_list, nlp):
    example_list = list()
    for doc_id in id_list:
        print("Processing :" + doc_id)
        source_dict = generate_source(file_path+doc_id)
        json_list = generate_json(source_dict, nlp)
        example_list.extend(json_list)
    return example_list

def generate_source(file_path):
    key_words = ["DOC","DATETIME","BODY","HEADLINE","TEXT"]
    source_dict = dict()
    doc_file = codecs.open(file_path+'.sgm', 'r', 'utf-8')
    doc_content = "".join(doc_file.readlines())
    doc_file.close()
    doc_content = doc_content.replace('\n',' ').replace('&amp;','&')
    pattern = re.compile('\<.*?\>')
    doc_content = re.sub(pattern, '', doc_content)
    source_dict['doc_content'] = doc_content
    try:
        tree = ET.parse(file_path+".apf.xml")
        root = tree.getroot()
    except Exception as e:
        print ("parse failed!")
        print(file_path+".apf.xml")
        sys.exit()
    source_dict = process_xml(source_dict, root)
    return source_dict

def process_xml(source_dict, root):
    entityList = root[0].findall("entity")
    entity_dict = dict()
    for entity in entityList:
        entityType = entity.attrib['TYPE']
        entitySubType = entity.attrib['SUBTYPE']
        entity_dict = get_entity_mentions(entity_dict, entity, entityType+":"+entitySubType, source_dict['doc_content'])
    timeList = root[0].findall("timex2")
    for time in timeList:
        entity_dict =  get_time_mentions(entity_dict, time, source_dict['doc_content'])
    valueList = root[0].findall("value")
    for value in valueList:
        entity_dict = get_value_mentions(entity_dict, value, source_dict['doc_content'])   
    source_dict['entity_dict'] = entity_dict

    eventList = root[0].findall("event")
    eventMentionList = list()
    for event in eventList:
        eventType = event.attrib['TYPE']
        eventSubtype = event.attrib['SUBTYPE']
        eventMentionList.extend(
            get_event_mentions(event, eventType+":"+eventSubtype, source_dict['doc_content']))
    source_dict['event_list'] = eventMentionList
    return source_dict

def get_entity_mentions(entity_dict, entity, entityType, doc_content):
    for mention in entity.findall('entity_mention'):
        mention_obj = {}
        mention_obj['entity_type'] = entityType
        extent = mention.find('extent').find('charseq')
        mention_obj['extent_start'] = int(extent.attrib['START'])
        mention_obj['extent_end'] = int(extent.attrib['END'])
        mention_obj['extent_content'] = extent.text.replace('\n',' ')
        cnt = mention_obj['extent_content'].count('&')
        if doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] != \
            mention_obj['extent_content']:
            mention_obj['extent_end'] -= 4*cnt
        while doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] != \
            mention_obj['extent_content']:
            mention_obj['extent_start'] -= 4
            mention_obj['extent_end'] -= 4
            if mention_obj['extent_start'] < 0:
                print(doc_content)
                print(doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1])
                print(mention_obj['extent_content'])
                print(doc_content.find(mention_obj['extent_content'], 1650))
                print(int(extent.attrib['START']))
                break
        assert doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] == mention_obj['extent_content']
        head = mention.find('head').find('charseq')
        mention_obj['head_start'] = int(head.attrib['START'])
        mention_obj['head_end'] = int(head.attrib['END'])
        mention_obj['head_content'] = head.text.replace('\n',' ')
        cnt = mention_obj['head_content'].count('&')
        if doc_content[mention_obj['head_start']:mention_obj['head_end']+1] != \
            mention_obj['head_content']:
            mention_obj['head_end'] -= 4*cnt
        while doc_content[mention_obj['head_start']:mention_obj['head_end']+1] != \
            mention_obj['head_content']:
            mention_obj['head_start'] -= 4
            mention_obj['head_end'] -= 4
        assert doc_content[mention_obj['head_start']:mention_obj['head_end']+1] == mention_obj['head_content']
        entity_dict[mention.attrib['ID']] = mention_obj
    return entity_dict

def get_time_mentions(entity_dict, entity, doc_content):
    for mention in entity.findall('timex2_mention'):
        mention_obj = {}
        mention_obj['entity_type'] = "TIME:TIME"
        extent = mention.find('extent').find('charseq')
        mention_obj['extent_start'] = int(extent.attrib['START'])
        mention_obj['extent_end'] = int(extent.attrib['END'])
        mention_obj['extent_content'] = extent.text.replace('\n',' ')
        cnt = mention_obj['extent_content'].count('&')
        mention_obj['extent_end'] -= 4*cnt
        while doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] != \
            mention_obj['extent_content']:
            mention_obj['extent_start'] -= 4
            mention_obj['extent_end'] -= 4
        mention_obj['head_start'] = mention_obj['extent_start']
        mention_obj['head_end'] = mention_obj['extent_end']
        mention_obj['head_content'] = mention_obj['extent_content']
        assert doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] == mention_obj['extent_content']
        entity_dict[mention.attrib['ID']] = mention_obj
    return entity_dict

def get_value_mentions(entity_dict, entity, doc_content):
    for mention in entity.findall('value_mention'):
        mention_obj = {}
        mention_obj['entity_type'] = "VALUE:VALUE"
        extent = mention.find('extent').find('charseq')
        mention_obj['extent_start'] = int(extent.attrib['START'])
        mention_obj['extent_end'] = int(extent.attrib['END'])
        mention_obj['extent_content'] = extent.text.replace('\n',' ')
        cnt = mention_obj['extent_content'].count('&')
        mention_obj['extent_end'] -= 4*cnt
        while doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] != \
            mention_obj['extent_content']:
            mention_obj['extent_start'] -= 4
            mention_obj['extent_end'] -= 4
        mention_obj['head_start'] = mention_obj['extent_start']
        mention_obj['head_end'] = mention_obj['extent_end']
        mention_obj['head_content'] = mention_obj['extent_content']
        assert doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] == mention_obj['extent_content']
        entity_dict[mention.attrib['ID']] = mention_obj
    return entity_dict

def get_event_mentions(event, eventType, doc_content):
    mentionList = []
    for mention in event.findall('event_mention'):
        mention_obj = {}
        mention_obj['event_mention_id'] = mention.attrib['ID']
        mention_obj['event_type'] = eventType
        extent = mention.find('extent').find('charseq')
        mention_obj['extent_start'] = int(extent.attrib['START'])
        mention_obj['extent_end'] = int(extent.attrib['END'])
        mention_obj['extent_content'] = extent.text.replace('\n',' ')
        cnt = mention_obj['extent_content'].count('&')
        mention_obj['extent_end'] -= 4*cnt
        while doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] != \
            mention_obj['extent_content']:
            mention_obj['extent_start'] -= 4
            mention_obj['extent_end'] -= 4
        assert doc_content[mention_obj['extent_start']:mention_obj['extent_end']+1] == mention_obj['extent_content']
        anchor = mention.find('anchor').find('charseq')
        mention_obj['anchor_start'] = int(anchor.attrib['START'])
        mention_obj['anchor_end'] = int(anchor.attrib['END'])
        mention_obj['anchor_content'] = anchor.text.replace('\n',' ').replace('\n',' ')
        cnt = mention_obj['anchor_content'].count('&')
        mention_obj['anchor_end'] -= 4*cnt
        while doc_content[mention_obj['anchor_start']:mention_obj['anchor_end']+1] != \
            mention_obj['anchor_content']:
            mention_obj['anchor_start'] -= 4
            mention_obj['anchor_end'] -= 4
        assert doc_content[mention_obj['anchor_start']:mention_obj['anchor_end']+1] == mention_obj['anchor_content']
        mention_obj['argument_list'] = get_argument(mention)
        mentionList.append(mention_obj)
    return mentionList

def get_argument(event_mention):
    argumentList = []
    for argument in event_mention.findall('event_mention_argument'):
        argument_obj = {}
        argument_obj['ref_id'] = argument.attrib['REFID']
        argument_obj['role'] = argument.attrib['ROLE']
        argumentList.append(argument_obj)
    return argumentList


def generate_json(source_dict, nlp):
    example_list = list()
    doc_token, doc_span = nlp.sentence_tokenize(source_dict['doc_content'], span=True)
    doc_pre = 0
    while source_dict['doc_content'][doc_pre] == ' ':
        doc_pre = doc_pre + 1
    for sentence_token, sentence_span in zip(doc_token, doc_span):
        # each sentence
        sen_obj = dict()
        # ipdb.set_trace()
        sen_obj["sentence_start"] = doc_pre + sentence_span[0][0]
        sen_obj["sentence_end"] = doc_pre + sentence_span[-1][-1]
        sen_obj['sentence'] = source_dict['doc_content'][sen_obj["sentence_start"]:sen_obj["sentence_end"]]
        # print(sen_obj['sentence'])
        sen_obj['words'] = list()
        sen_obj['index'] = list()
        # ipdb.set_trace()
        word_list = nlp.word_tokenize(sen_obj['sentence'])
        pre = 0
        for word in word_list:
            # ipdb.set_trace()
            sen_obj['words'].append(word)
            bg = sen_obj['sentence'].find(word, pre)
            ed = bg + len(word) - 1
            pre = ed + 1
            assert sen_obj['sentence'][bg:ed + 1] == word
            sen_obj['index'].append((bg+sen_obj['sentence_start'], ed+sen_obj['sentence_start']))
        sen_obj['pos-tags'] = list()
        pos_list = nlp.pos_tag(sen_obj['sentence'])
        for word, (_word, pos) in zip(sen_obj['words'], pos_list):
            assert word == _word
            sen_obj['pos-tags'].append(pos)
        assert len(sen_obj['words']) == len(sen_obj['pos-tags'])
        dependency_list = nlp.dependency_parse(sen_obj['sentence'])
        sen_obj['stanford-colcc'] = list()
        for relation, st, ed in dependency_list:
            sen_obj['stanford-colcc'].append(
                relation + "/dep=" + str(ed - 1) + "/gov=" + str(st - 1))
        # ipdb.set_trace()
        sen_obj['golden-event-mentions'] = list()
        sen_obj['golden-entity-mentions'] = list()
        for entity_id in source_dict['entity_dict']:
            entity_mention = source_dict['entity_dict'][entity_id]
            entity_mention['index'] = None
            # sentence fully contains entity
            if entity_mention['head_start'] >= sen_obj["sentence_start"] and \
                entity_mention['head_end'] <= sen_obj['sentence_end']:
                entity_obj = dict()
                entity_obj['entity-type'] = entity_mention['entity_type']
                e_st = len(sen_obj['words']) - 1; e_ed = 0
                for idx, span in enumerate(sen_obj['index']):
                    # inter section between entity and word
                    if max(span[0], entity_mention['head_start']) <= min(span[1], entity_mention['head_end']):
                        e_st = min(e_st, idx); e_ed = max(e_ed, idx);
                entity_obj['start'] = e_st; entity_obj['end'] = e_ed+1
                entity_mention['index'] = (e_st, e_ed+1)
                sen_obj['golden-entity-mentions'].append(entity_obj)
                if " ".join(sen_obj['words'][e_st:e_ed+1]) != entity_mention['head_content']:
                    print("entity not exactly matched")
                    print(" ".join(sen_obj['words'][e_st:e_ed+1]))
                    print(entity_mention['head_content'])
                    # ipdb.set_trace()
        for event_mention in source_dict['event_list']:
            if event_mention['anchor_start'] >= sen_obj["sentence_start"] and \
                event_mention['anchor_end'] <= sen_obj['sentence_end']:
                event_obj = dict()
                event_obj['event_type'] = event_mention['event_type']
                e_st = len(sen_obj['words']) - 1; e_ed = 0
                for idx, span in enumerate(sen_obj['index']):
                    # inter section between entity and word
                    if max(span[0], event_mention['anchor_start']) <= min(span[1], event_mention['anchor_end']):
                        e_st = min(e_st, idx); e_ed = max(e_ed, idx);
                if " ".join(sen_obj['words'][e_st:e_ed+1]) != event_mention['anchor_content']:
                    print("Trigger not exactly matched")
                    print(" ".join(sen_obj['words'][e_st:e_ed+1]))
                    print(event_mention['anchor_content'])
                event_obj['trigger'] = dict()
                event_obj['trigger']['start'] = e_st; event_obj['trigger']['end'] = e_ed+1
                event_obj['trigger']['text'] = event_mention['anchor_content']
                event_obj['arguments'] = list()
                for argument_obj in event_mention['argument_list']:
                    ref_id = argument_obj['ref_id']
                    if source_dict['entity_dict'][ref_id]['index'] is not None:
                        event_obj['arguments'].append({
                            'start': source_dict['entity_dict'][ref_id]['index'][0],
                            'end': source_dict['entity_dict'][ref_id]['index'][1],
                            'text': source_dict['entity_dict'][ref_id]['head_content'],
                            'role': argument_obj['role']
                            })
                    else:
                        print("Argument not found: ")
                        print(source_dict['entity_dict'][ref_id])
                        print("At Trigger: ")
                        print(event_obj['trigger'])
                        # ipdb.set_trace()
                sen_obj['golden-event-mentions'].append(event_obj)
        example_list.append(sen_obj)
    return example_list

def main():
    nlp = StanfordCoreNLP(
        "stanford-corenlp-full-2018-10-05")
    train_id, dev_id, test_id = load_split_file()

    json_file_path = "ace_2005_td_v7/data/English/"
    train_list = generate_example_list(json_file_path, train_id, nlp)
    print("train ", len(train_list))
    dev_list = generate_example_list(json_file_path, dev_id, nlp)
    print("dev ", len(dev_list))
    test_list = generate_example_list(json_file_path, test_id, nlp)
    print("test ", len(test_list))

    _file = codecs.open("JMEE_train.json", 'w', 'utf-8')
    json.dump(train_list, _file, indent=2)
    _file.close()
    _file = codecs.open("JMEE_dev.json", 'w', 'utf-8')
    json.dump(dev_list, _file, indent=2)
    _file.close()
    _file = codecs.open("JMEE_test.json", 'w', 'utf-8')
    json.dump(test_list, _file, indent=2)
    _file.close()


def test():
    nlp = StanfordCoreNLP(
        "stanford-corenlp-full-2018-10-05")
    source_dict = generate_source("ace_2005_td_v7/data/English/bc/timex2norm/CNN_CF_20030303.1900.00")
    json_list = generate_json(source_dict, nlp)

if __name__ == "__main__":
    main()
    # test()