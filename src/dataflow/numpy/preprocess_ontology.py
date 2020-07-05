import ujson as json
from collections import defaultdict

# generate ontology graph

def ont_ace_by_json(json_file, coarse_type=True):
    parsed_data = json.load(open(json_file))

    entity_type_set = set() # 'WEA', 'ORG', 'GPE', 'LOC', 'TIME', 'VEH', 'VALUE', 'FAC', 'PER'
    role_set = defaultdict(int)
    ont_dict = defaultdict(lambda : defaultdict(set))
    for sent in parsed_data:
        entities = sent['golden-entity-mentions']
        entity_dict = defaultdict(lambda: defaultdict(list))
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            if coarse_type:
                entity_type = entity['entity-type'].split(':')[0]
            else:
                entity_type = entity['entity-type']
            entity_type_set.add(entity_type)
            entity_dict[entity_start][entity_end] = entity_type
        events = sent['golden-event-mentions']
        for event in events:
            event_type = event['event_type']
            args = event['arguments']
            for arg in args:
                # role = '%s_%s' % (event_type, arg['role'])
                role = arg['role']
                # role_set.add(role)
                role_set[role] += 1
                arg_start = arg['start']
                arg_end = arg['end']
                ont_dict[event_type][role].add(entity_dict[arg_start][arg_end])

    print(entity_type_set)
    # generate role idx for constant.py
    idx = 1
    dict = {}
    for role in role_set:
        dict[role] = idx
        idx = idx + 1
    print(dict)
    return ont_dict

if __name__ == '__main__':
    json_file = '/scratch/manling2/data/mm-event-graph/ace/JMEE_train.json'
    out_file = '/scratch/manling2/data/mm-event-graph/ace/ontology.json'
    ont_dict = ont_ace_by_json(json_file, coarse_type=False)
    # print(ont_dict)
    json.dump(ont_dict, open(out_file, 'w'), indent=4)