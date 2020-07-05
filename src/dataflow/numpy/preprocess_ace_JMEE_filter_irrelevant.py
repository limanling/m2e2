import os
import json
import sys
sys.path.append('../../..')
from src.dataflow.numpy.anno_mapping import event_type_norm, role_name_norm
from collections import defaultdict

# for name in ['train', 'test', 'dev']:
#     filename = '/scratch/manling2/data/mm-event-graph/ace/JMEE_%s.json' % name
#     filename_new = '/scratch/manling2/data/mm-event-graph/ace/JMEE_%s_filter.json' % name

filename = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/article.json'
filename_new = '/scratch/manling2/mm-event-graph/data/voa_anno_m2e2/article_filter.json'

sr_ace_mapping = '../../../data/ace/ace_sr_mapping.txt'

ace_events = set()
ace_roles = set()
for line in open(sr_ace_mapping):
    line = line.rstrip('\n')
    tabs = line.split('\t')
    sr_verb = tabs[0]
    sr_role = tabs[1]
    ee_event = event_type_norm(tabs[2])
    ee_role = tabs[3]
    ace_events.add(ee_event)
    # ace_roles.add()
print(ace_events)

data = json.load(open(filename))

not_in = set()
keep = set()

data_new = list()
# for sent in data:
#     for event in sent['golden-event-mentions']:
#         print('type', event['event_type'])
#         if event_type_norm(event['event_type']) in ace_events:
#             data_new.append(sent)
#             keep.add(event['event_type'])
#             break
#         else:
#             not_in.add(event['event_type'])
for sent in data:
    sent_new = dict()
    deleted_entities = defaultdict(lambda : defaultdict(str))
    for key in sent:
        if key == 'golden-event-mentions':
            sent_new['golden-event-mentions'] = list()
            # remove arguments
            for event in sent['golden-event-mentions']:
                if event_type_norm(event['event_type']) in ace_events:
                    sent_new['golden-event-mentions'].append(event)
                    keep.add(event['event_type'])
                else:
                    not_in.add(event['event_type'])
        else:
            sent_new[key] = sent[key]

    data_new.append(sent_new)

json.dump(data_new, open(filename_new, 'w'), indent=4)




print('remove ', str(not_in))
print('keep', str(keep))