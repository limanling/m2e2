from collections import defaultdict
# import ujson as json
import json
'''
filter out time/value
'''

# file_types = ['train', 'test', 'dev', 'train_filter', 'test_filter', 'dev_filter']
# for file_type in file_types:
#     filename = '/scratch/manling2/data/mm-event-graph/ace/JMEE_%s.json' % file_type
#     filename_new = '/scratch/manling2/data/mm-event-graph/ace/JMEE_%s_no_time.json' % file_type

filename = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/article.json'
filename_new = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/article_filter.json'

data = json.load(open(filename))

not_in = set()
keep = set()

data_new = list()
for sent in data:
    sent_new = dict()
    deleted_entities = defaultdict(lambda : defaultdict(str))
    for key in sent:
        if key == 'golden-entity-mentions':
            sent_new['golden-entity-mentions'] = list()
            for entity in sent['golden-entity-mentions']:
                entity_type = entity['entity-type']
                entity_start = entity['start']
                entity_end = entity['end']
                # print('type', entity_type)
                # if entity_type.startswith('TIME'):
                if entity_type.startswith('TIME') or entity_type.startswith('VALUE'):
                    deleted_entities[entity_start][entity_end] = entity_type
                else:
                    sent_new['golden-entity-mentions'].append(entity)
        elif key == 'golden-event-mentions':
            sent_new['golden-event-mentions'] = list()
            # remove arguments
            for event in sent['golden-event-mentions']:
                event_new = dict()
                event_new['trigger'] = event['trigger']
                event_new['arguments'] = list()
                for arg in event['arguments']:
                    arg_start = arg['start']
                    arg_end = arg['end']
                    if arg_start in deleted_entities and arg_end in deleted_entities[arg_start]:
                        continue
                    else:
                        event_new['arguments'].append(arg)
                event_new['event_type'] = event['event_type']
                sent_new['golden-event-mentions'].append(event_new)
        else:
            sent_new[key] = sent[key]

    data_new.append(sent_new)


json.dump(data_new, open(filename_new, 'w'), indent=4)


    # how to remove pronoun??