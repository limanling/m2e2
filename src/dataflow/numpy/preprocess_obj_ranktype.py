'''
rank the object types based on frequency, to check which one should be avoided
'''

import pickle
from collections import defaultdict

import sys
sys.path.append('../../../')
from src.dataflow.numpy.data_loader_situation import get_labels

obj_file = '/scratch/manling2/mm-event-graph/data/voa/object_detection/det_results_voa_oi_1.pkl'
img_objects = pickle.load(open(obj_file), 'rb')

class_map_file = '/scratch/manling2/mm-event-graph/data/voa/class-descriptions-boxable.csv'
object_label = get_labels(class_map_file)

label_count = defaultdict()
for img in img_objects:
    for object in img_objects[img]:
        label = object_label[object['label']]
        label_count[label] += 1

for label, count in sorted(label_count.items(), key=lambda x:x[1], reverse=True):
    print(label, count)

