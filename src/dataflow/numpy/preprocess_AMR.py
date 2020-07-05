# import ujson as json
import json

core_roles = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
              'ARG0-of', 'ARG1-of', 'ARG2-of', 'ARG3-of', 'ARG4-of',
              'mod', 'location', 'instrument', 'poss', 'manner', 'topic', 'medium', 'prep-X',
              'year', 'duration', 'decade', 'weekday', 'time',
              'destination', 'path', 'location']

def generate_sents(json_dep, sents):
    data = json.load(open(json_dep))
    writer = open(sents, 'w')
    for sent in data:
        writer.write(sent['sentence'])
        writer.write('\n')
    writer.flush()
    writer.close()

def read_AMR(amr_result):
    # read AMR results -> head, edge
    parent_each_level = dict()
    head = dict()
    edge = dict()
    for line in open(amr_result):
        line = line.rstrip('\n')
        # print(line)
        if line.startswith('# ::id'):
            prefix_len = len('# ::id ')
            sent_idx = int(line[prefix_len:])
            parent_each_level[sent_idx] = dict()
            head[sent_idx] = dict()
            edge[sent_idx] = dict()
            # if sent_idx - 1 in parent_each_level:
            #     # print(sent_idx - 1, parent_each_level[sent_idx-1])
            #     print(sent_idx - 1, head[sent_idx - 1])
            #     print(sent_idx - 1, edge[sent_idx - 1])
            #     print()
            #     # break
        # elif not line.startswith('# ::snt'):
        elif line.startswith('(x / xconcept'):
            # two root verbs
            parent_each_level[sent_idx][0] = -1
        elif line.startswith('(xap'):
            # anonmyous root verbs
            parent_each_level[sent_idx][0] = -1
            # root_id = int(line[len('(xap'):line.find(' /')])
            # parent_each_level[sent_idx][0] = root_id
            # head[sent_idx][root_id] = -1
            # edge[sent_idx][root_id] = 'ROOT'
            # word_idx = root_id
        elif line.startswith('(x'):
            # the root verb of AMR graph
            root_id = int(line[len('(x'):line.find(' /')])
            parent_each_level[sent_idx][0] = root_id
            head[sent_idx][root_id] = -1
            edge[sent_idx][root_id] = 'ROOT'
            word_idx = root_id  # start from 1
        elif line.startswith('\t'):
            line_start = line.find(':')
            prefix = line[:line_start]
            content = line[line_start:]
            depth = len(prefix)
            label = content[:content.find(' ')]
            if label == ':x':
                label = 'ROOT'
            # print('label', label)
            if label.startswith(':op') and ' "' in content:
                op_idx = int(content[len(':op'): content.find(' ')])
                head[sent_idx][word_idx + op_idx - 1] = head[sent_idx][word_idx]
                edge[sent_idx][word_idx + op_idx - 1] = edge[sent_idx][word_idx]
            # elif ' (xap' in content:
            #     word_idx = int(content[content.find('(xap') + len('(xap'): content.find(' /')])
            #     parent_each_level[sent_idx][depth] = word_idx
            #     head[sent_idx][word_idx] = parent_each_level[sent_idx][depth - 1]
            #     edge[sent_idx][word_idx] = label
            elif ' (x' in content and ' (xap' not in content:
                word_idx = int(content[content.find('(x') + len('(x'): content.find(' /')])
                parent_each_level[sent_idx][depth] = word_idx
                head[sent_idx][word_idx] = parent_each_level[sent_idx][depth - 1]
                edge[sent_idx][word_idx] = label
            else:
                # '(n' use the last '(x'
                parent_each_level[sent_idx][depth] = word_idx
                # print('No word line', label)
        # print(parent_each_level[sent_idx], head[sent_idx])
    return head, edge

if __name__ == "__main__":

    dataset_list = ['train', 'test', 'dev']
    # dataset_list = ['test']

    # generate input sentence file with one sent each line
    # for dataset in dataset_list:
    #     json_dep = '/dvmm-filer2/users/manling/mm-event-graph2/data/ace/JMEE_%s.json' % dataset
    #     sents = '/dvmm-filer2/users/manling/mm-event-graph2/data/ace/sents_%s.txt' % dataset
    #     generate_sents(json_dep, sents)

    # run AMR parser on this dataset
    # cd /dvmm-filer2/users/manling/AMRParsing
    # python amr_parsing.py -m preprocess /dvmm-filer2/users/manling/mm-event-graph2/data/ace/sents_test.txt
    # python amr_parsing.py -m parse --model /dvmm-filer2/users/manling/AMRParsing/models/semeval/amr-semeval-all.train.basic-abt-brown-verb.m /dvmm-filer2/users/manling/mm-event-graph2/data/ace/sents_test.txt 2>log/error.log

    edge_types = set()
    for dataset in dataset_list:
        json_dep = '/dvmm-filer2/users/manling/mm-event-graph2/data/ace/JMEE_%s.json' % dataset
        amr_result = '/dvmm-filer2/users/manling/mm-event-graph2/data/ace/sents_%s.txt.all.basic-abt-brown-verb.parsed' % dataset
        amr_tok = '/dvmm-filer2/users/manling/mm-event-graph2/data/ace/sents_%s.txt.prp' % dataset
        head, edge = read_AMR(amr_result)

        # get word index mapping
        # amr_tokens = dict()
        # sent_id = 0
        # for line in open(amr_tok):
        #     line = line.rstrip('\n')
        #     if line.startswith('Sentence #'):
        #         sent_id = sent_id + 1
        #         token_id = 0
        #         amr_tokens[sent_id] = dict()
        #     elif line.startswith('[Text='):
        #         token_id = token_id + 1
        #         content_start = line[line.find('CharacterOffsetBegin='):]
        #         token_start = int(content_start[len('CharacterOffsetBegin='):content_start.find(' ')])
        #         content_end = line[line.find('CharacterOffsetEnd='):]
        #         token_end = int(content_end[len('CharacterOffsetEnd='):content_end.find(' ')])
        #         amr_tokens[sent_id][token_id] = (token_start, token_end)


        # add to the json files
        data = json.load(open(json_dep))
        for sent_idx, sent_dict in enumerate(data):
            sent_idx = sent_idx + 1
            sent_dict['sent_idx'] = sent_idx
            sent_dict['amr-colcc'] = list()

            # word2token = dict()
            # for token_idx, (token_start, token_end) in enumerate(sent_dict['index']):
            #     print('gt_tokens', token_idx, token_start, token_end)
            #     print('amr_tokens', amr_tokens[sent_idx])
            #     for amr_word_idx, (amr_start, amr_end) in enumerate(amr_tokens[sent_idx]):
            #         if int(amr_start) >= int(token_start) and int(amr_end) <= int(token_end)+1:
            #             word2token[int(amr_word_idx)] = int(token_idx)
            # print('word2token', word2token)

            word_len = len(sent_dict['words'])
            for word_idx in head[sent_idx]:
                # map word_idx to token_idx:
                # token_idx = word2token[word_idx]
                # dep_idx = word2token[word_idx]
                # if head[sent_idx][word_idx] == -1:
                #     gov_idx = -1
                # else:
                #     gov_idx = word2token[head[sent_idx][word_idx]]
                # word_idx from 1, should change to start from 0
                if word_idx >= word_len:
                    print('WRONG ', sent_idx, word_idx, word_len)
                if head[sent_idx][word_idx] != -1:
                    gov_idx = head[sent_idx][word_idx] - 1
                else:
                    gov_idx = head[sent_idx][word_idx]  # = -1
                dep_idx = word_idx - 1
                # add result
                if dep_idx <= word_len:
                    sent_dict['amr-colcc'].append('%s/dep=%d/gov=%d' % (edge[sent_idx][word_idx], dep_idx, gov_idx))
                edge_types.add(edge[sent_idx][word_idx])
            for word_idx in range(word_len):
                if word_idx not in head[sent_idx]:
                    sent_dict['amr-colcc'].append(
                        '%s/dep=%d/gov=%d' % ('ROOT', word_idx, -1))
        json.dump(data, open(json_dep.replace('.json', '_amr.json'), 'w'), indent=4)
    print('edge type number', len(edge_types))
