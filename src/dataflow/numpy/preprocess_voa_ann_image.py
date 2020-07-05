'''
Input are xmls
'''

from collections import defaultdict
import codecs
import json
import sys
sys.path.append('../../../')
from src.dataflow.annotation.visualize import read_image_ent_file, id2event
from src.dataflow.numpy.anno_mapping import event_type_mapping_image2ace, event_role_mapping_image2ace
import os

image_type_role_dir = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/image_all_final/batch_all/image_entity'
image_json = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/image_event.json'
text_ann = '/scratch/manling2/data/mm-event-graph/voa_anno_m2e2/text_all/article_ann' # delete the image that has no text ann

image_set = defaultdict(lambda: defaultdict())
doc_set = defaultdict(set)
for image_xml in os.listdir(image_type_role_dir):
    if not image_xml.endswith('.xml'):
        continue
    # print(image_xml)
    ents = read_image_ent_file(os.path.join(image_type_role_dir, image_xml))  # [[event_type_id, arg_role, name, xmin, ymin, xmax, ymax]]

    image_id_nosuffix = image_xml.replace('.xml', '')
    doc_id = image_id_nosuffix[:image_id_nosuffix.rfind('_')]

    # delete the image that has no text ann
    doc_ann_path = os.path.join(text_ann, doc_id+'.rsd.ann')
    if not os.path.exists(doc_ann_path):
        print('[ERROR]NoANN %s' % doc_id)
        continue

    doc_set[doc_id].add(image_id_nosuffix)

    for ent in ents:
        event_type_raw = id2event[ent[0]].strip(' ')
        print('raw', event_type_raw)
        event_type = event_type_mapping_image2ace[event_type_raw]
        print('clean', event_type)
        role = ent[1].strip(' ').replace('demonstartor', 'demonstrator')
        role = event_role_mapping_image2ace[event_type_raw][role]
        entity_name = ent[2].strip(' ')
        xmin = int(ent[3])
        ymin = int(ent[4])
        xmax = int(ent[5])
        ymax = int(ent[6])

        if 'role' not in image_set[image_id_nosuffix]:
            image_set[image_id_nosuffix]['role'] = defaultdict(list)
        image_set[image_id_nosuffix]['role'][role].append((entity_name, xmin, ymin, xmax, ymax))

    image_set[image_id_nosuffix]['event_type'] = event_type

image_json_writer = codecs.open(image_json, 'w', 'utf-8')
json.dump(image_set, image_json_writer, indent=2)