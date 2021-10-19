import json
from collections import defaultdict
import os
import sys
sys.path.append('../../')
from src.dataflow.numpy.anno_mapping import entity_type_mapping_brat, event_type_mapping_ace2brat, event_role_mapping_ace2brat, event_role_mapping_brat2ace
import shutil

class EDVisualizer():
    def __init__(self, JMEE_data_json):
        self.sent_info = defaultdict(lambda : defaultdict())
        self._load_sent_info(JMEE_data_json)

    def _load_sent_info(self, JMEE_data_json):
        raw_data = json.load(open(JMEE_data_json))
        for data_instance in raw_data:
            sent_id = data_instance['sentence_id']
            words = data_instance['words']
            # index_all = data_instance['index_all']
            # index = data_instance['index']
            # entity = data_instance['golden-entity-mentions']
            sentence = data_instance['sentence']

            self.sent_info[sent_id]['words'] = words
            # self.sent_info[sent_id]['index_all'] = index_all
            # self.sent_info[sent_id]['index'] = index
            # self.sent_info[sent_id]['entity'] = entity
            self.sent_info[sent_id]['sentence'] = sentence


    def save_json(self, predicted_event_triggers, predicted_events, sent_id, ee_role_i2s, text_result_json):
        sent_id = sent_id[0]
        for result_dict in predicted_events:
            for key in result_dict:
                # print('self.sent_info[sent_id]', self.sent_info)
                # print('sent_id', sent_id)
                eventstart_sentid, event_end, event_type_ace = key
                trigger_word = ' '.join(self.sent_info[sent_id]['words'][int(eventstart_sentid):int(event_end)])
                text_result_json[eventstart_sentid]['sentence_id'] = sent_id
                text_result_json[eventstart_sentid]['sentence'] = self.sent_info[sent_id]['sentence']
                text_result_json[eventstart_sentid]['sentence_tokens'] = self.sent_info[sent_id]['words']
                text_result_json[eventstart_sentid]['pred_event_type'] = event_type_ace
                text_result_json[eventstart_sentid]['pred_trigger'] = {'index_start': eventstart_sentid, 'index_end':event_end, 'text':trigger_word}
                text_result_json[eventstart_sentid]['pred_roles'] = defaultdict(list)

                args = result_dict[key]
                for arg_start, arg_end, arg_type in args:
                    arg_word = ' '.join(self.sent_info[sent_id]['words'][int(arg_start):int(arg_end)])
                    # arg_start_offset, arg_end_offset = self.get_offset_by_idx(sent_id, arg_start, arg_end)
                    role_name_ace = ee_role_i2s[arg_type]
                    text_result_json[eventstart_sentid]['pred_roles'][role_name_ace].append( {'index_start':arg_start, 'index_end':arg_end, 'text':arg_word} )



    def visualize_html(self, predicted_event_triggers, predicted_events, sent_id, ee_role_i2s, visual_writer):
        for eventstart_sentid in predicted_event_triggers:
            # print('eventstart_sentid', eventstart_sentid)
            event_end, event_type_ace = predicted_event_triggers[eventstart_sentid]
            # if '__' in eventstart_sentid:
            event_start = int(eventstart_sentid[eventstart_sentid.find('__')+2:])
            # print('predicted_event_triggers', predicted_event_triggers)
            # event_start_offset, event_end_offset = self.get_offset_by_idx(sent_id, event_start, event_end)
            visual_writer.write("sentence: %s<br>\n" % self.sent_info[sent_id]['sentence'])
            trigger_word = ' '.join(self.sent_info[sent_id]['words'][event_start:event_end])  # multiple words]
            visual_writer.write("event type: %s<br>\n" % event_type_ace)
            visual_writer.write("trigger word: %s<br>\n" % trigger_word)
            if (eventstart_sentid, event_end, event_type_ace) in predicted_events:
                args = predicted_events[(eventstart_sentid, event_end, event_type_ace)]
                for arg_start, arg_end, arg_type in args:
                    arg_word = ' '.join(self.sent_info[sent_id]['words'][arg_start:arg_end])
                    # arg_start_offset, arg_end_offset = self.get_offset_by_idx(sent_id, arg_start, arg_end)
                    role_name_ace = ee_role_i2s[arg_type]
                    visual_writer.write("        [%s]: %s<br>\n" % (role_name_ace, arg_word))
            visual_writer.write('<br>\n<br>\n')

        visual_writer.flush()
        visual_writer.close()


    def visualize_brat(self, predicted_event_triggers, predicted_events, sent_id, ee_role_i2s, visual_writer, save_entity=False):
        print('predicted_event_triggers', predicted_event_triggers)
        print('predicted_events', predicted_events)
        for eventstart_sentid in predicted_event_triggers:
            # print('eventstart_sentid', eventstart_sentid)
            event_end, event_type_ace = predicted_event_triggers[eventstart_sentid]
            # if '__' in eventstart_sentid:
            event_start = int(eventstart_sentid[eventstart_sentid.find('__')+2:])
            # print('predicted_event_triggers', predicted_event_triggers)
            event_start_offset, event_end_offset = self.get_offset_by_idx(sent_id, event_start, event_end)
            trigger_word = ' '.join(self.sent_info[sent_id]['words'][event_start:event_end])  # multiple words
            # T44	Attack 6595 6604	onslaught
            # event_id = 'T%s_%d_%d' % (sent_id[sent_id.find('_'):], event_start_offset, event_end_offset)
            token_id = 'T%d_%d' % (event_start_offset, event_end_offset)
            if event_type_ace not in event_type_mapping_ace2brat:
                print('ignored type', event_type_ace)
                continue
            event_type_brat = event_type_mapping_ace2brat[event_type_ace]
            visual_writer.write(token_id)
            visual_writer.write('\t%s' % event_type_brat)
            visual_writer.write(' %d %d' % (event_start_offset, event_end_offset))
            visual_writer.write('\t%s\n' % trigger_word)

            # E11	Attack:T44 Attacker:T26493 Target:T45
            # save triggers
            event_id = token_id.replace('T', 'E')
            visual_writer.write('%s' % event_id)
            visual_writer.write('\t%s:%s' % (event_type_brat, token_id))
            # save args
            if save_entity:
                entity_lines = list()
            # try:
            if (eventstart_sentid, event_end, event_type_ace) in predicted_events:
                args = predicted_events[(eventstart_sentid, event_end, event_type_ace)]
                for arg_start, arg_end, arg_type in args:
                    # index -> offset index
                    arg_word = ' '.join(self.sent_info[sent_id]['words'][arg_start:arg_end])
                    arg_start_offset, arg_end_offset = self.get_offset_by_idx(sent_id, arg_start, arg_end)
                    role_name_ace = ee_role_i2s[arg_type]
                    # print(event_type_ace, role_name_ace)
                    role_name_brat = event_role_mapping_ace2brat[event_type_ace][role_name_ace]
                    visual_writer.write(' %s:T%d_%d' % (role_name_brat, arg_start_offset, arg_end_offset))
                    if save_entity:
                        # T44	Attack 6595 6604	onslaught
                        arg_type_brat = 'VAL'  # do not save the entity types?
                        entity_lines.append('T%d_%d\t%s %d %d\t%s\n' % (
                            arg_start_offset, arg_end_offset, arg_type_brat,
                            arg_start_offset, arg_end_offset, arg_word
                        ))
            # except:
            #     print('no arguments ', (eventstart_sentid, event_end, event_type_ace))
            visual_writer.write('\n')
            if save_entity:
                visual_writer.write(''.join(entity_lines))

        visual_writer.flush()
        visual_writer.close()

    def get_offset_by_idx(self, sent_id, idx_start, idx_end):
        # print('idx_start', idx_start, 'idx_end', idx_end)
        # print(self.sent_info[sent_id]['index_all'])
        start_offset, _ = self.sent_info[sent_id]['index_all'][idx_start]
        _, end_offset = self.sent_info[sent_id]['index_all'][idx_end - 1]
        end_offset = end_offset + 1  # brat is [start, end] not [start, end)
        return start_offset, end_offset

    def rewrite_brat(self, event_tmp_dir, ann_dir, save_entity=False):
        '''
        The previous format is not brat format, need postprocessing
        :return:
        '''
        # if os.path.isdir(event_tmp_dir):
        for event_tmp_file in os.listdir(event_tmp_dir):
            if event_tmp_file.endswith('.ann_tmp'):
                event_tmp_path = os.path.join(event_tmp_dir, event_tmp_file)
                ann_path = os.path.join(ann_dir, event_tmp_file.replace('.ann_tmp', '.ann'))
                if not save_entity:
                    self._rewrite_brat(event_tmp_path, ann_path)
                else:
                    # remove repeated lines?
                    # simpler version: copy as final ann file, ignore the repeated lines
                    shutil.copyfile(event_tmp_path, event_tmp_path.replace('_tmp', ''))

                # copy rsd txt
                rsd_path = os.path.join(ann_dir, event_tmp_file.replace('.ann_tmp', '.txt'))
                rsd_path_new = os.path.join(event_tmp_dir, event_tmp_file.replace('.ann_tmp', '.txt'))
                shutil.copyfile(rsd_path, rsd_path_new)

    def _rewrite_brat(self, event_tmp_file, anno_file):
        # get all entity offset and entity id mapping
        token_offsetid2realid = dict()

        entity_lines = list()
        if os.path.exists(anno_file):
            lines = open(anno_file).readlines()
            for line in lines:
                if line.startswith('T'):
                    tabs = line.split('\t')
                    id = tabs[0]
                    subs = tabs[1].split(' ')
                    type = subs[0]
                    if type in entity_type_mapping_brat:
                        start = int(subs[1])
                        end = int(subs[2])
                        # mention = tabs[2]
                        offset_id = 'T%d_%d' % (start, end)
                        token_offsetid2realid[offset_id] = id
                        entity_lines.append(line)
        else:
            print('NoANN', anno_file)

        # write entities
        writer = open(event_tmp_file.replace('_tmp', ''), 'w')
        print(event_tmp_file.replace('_tmp', ''))
        writer.write(''.join(entity_lines))

        # write events:
        for line in open(event_tmp_file):
            line = line.rstrip('\n')
            if line.startswith('T'):
                writer.write('%s\n' % line)
            else:
                # update the arg token_id
                # E11	Attack:T44 Attacker:T26493 Target:T45
                role_values = line.split('\t')[-1].split(' ')
                # event_id = role_values[0].split(':')[1]
                # event_type = role_values[0].split(':')[0]
                writer.write(line.split('\t')[0])
                writer.write('\t')
                writer.write(role_values[0])
                entity_added_lines=list()
                for role_value in role_values[1:]:
                    # role_name_raw = role_value.split(':')[0].replace('2', '').replace('3', '').replace('4', '').replace(
                    #     '5', '')
                    # if len(role_name_raw) == 0:
                    #     continue
                    # if role_name_raw not in event_role_mapping_brat2ace[event_type]:
                    #     print('ignored role: ', event_id, role_value)
                    #     continue
                    # role_name = event_role_mapping_brat2ace[event_type][role_name_raw]
                    entity = role_value.split(':')[1]
                    if entity not in token_offsetid2realid:
                        print('[ERROR] entity id can not find in *.ann', entity, anno_file)
                        entity_type = 'VAL'
                        entity_str = 'VAL'
                        entity_added_lines.append('%s\t%s %s %s\t%s\n' % (entity, entity_type,
                                                             entity[1:].split('_')[0],
                                                             entity[1:].split('_')[1],
                                                             entity_str) )
                        entity_realid = entity
                    else:
                        entity_realid = token_offsetid2realid[entity]
                        # print('entity', entity, 'entity_realid', entity_realid)
                    writer.write(' ')
                    writer.write(role_value.replace(entity, entity_realid))
                writer.write('\n')
                writer.write(''.join(entity_added_lines))

        writer.flush()
        writer.close()

