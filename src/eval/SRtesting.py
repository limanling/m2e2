import os

class SRTester():
    def __init__(self):
        pass

    def calculate_lists(self, y, y_):
        '''
        for a sequence, whether the prediction is correct
        note that len(y) == len(y_)
        :param y:
        :param y_:
        :return:
        '''
        ct = 0
        p2 = len(y_)
        p1 = len(y)
        for i in range(p2):
            if y[i] == y_[i]:
                ct = ct + 1
        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_no_order(self, y, y_):
        '''
        for each predicted item, whether it is in the gt
        :param y: [batch, items]
        :param y_: [batch, items]
        :return:
        '''
        ct, p1, p2 = 0, 0, 0
        for batch, batch_ in zip(y, y_):
            value_set = set(batch)
            value_set_ = set(batch_)
            p1 += len(value_set)
            p2 += len(value_set_)

            for value_ in value_set_:
                # if value_ == '(0,0,0)':
                if value_ in value_set:
                    ct += 1

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_noun(self, y, y_):
        '''
        for each ground truth entity, whether it is in the predicted entities
        :param y: [batch, role_num, multiple_args]
        :param y_: [batch, role_num, multiple_entities]
        :return:
        '''
        # print('y', y)
        # print('y_', y_)
        ct, p1, p2 = 0, 0, 0
        # for batch_idx, batch_idx_ in zip(y, y_):
        #     batch = y[batch_idx]
        #     batch_ = y_[batch_idx_]
        for batch, batch_ in zip(y, y_):
            # print('batch', batch)
            # print('batch_', batch_)
            p1 += len(batch)
            p2 += len(batch_)
            for role in batch:
                found = False
                entities = batch[role]
                for entity in entities:
                    for role_ in batch_:
                        entities_ = batch_[role_]
                        if entity in entities_:
                            ct += 1
                            found = True
                            break
                    if found:
                        break

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_triple(self, y, y_):
        '''
        for each role, whether the predicted entities have overlap with the gt entities
        :param y: dict, role -> entities
        :param y_: dict, role -> entities
        :return:
        '''
        ct, p1, p2 = 0, 0, 0
        # for batch_idx, batch_idx_ in zip(y, y_):
        #     batch = y[batch_idx]
        #     batch_ = y_[batch_idx_]
        for batch, batch_ in zip(y, y_):
            p1 += len(batch)
            p2 += len(batch_)
            for role in batch:
                # is_correct = False
                entities = batch[role]
                if role in batch_:
                    entities_ = batch_[role]
                    for entity_ in entities_:
                        if entity_ in entities:
                            ct += 1
                            # is_correct = True
                            break
                # if not is_correct:
                #     print('Wrong one:', role, batch[role])

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1


    def visualize_sets_triple(self, image_id_batch, y_verb, y_verb_, y, y_, verb_id2s, role_id2s, noun_id2s, sr_visualpath, image_path=None):
        '''
        for each role, whether the predicted entities have overlap with the gt entities
        :param y: dict, role -> entities
        :param y_: dict, role -> entities
        :return:
        '''

        # sr_visualpath = '/scratch/manling2/html/m2e2/sr_errors'
        image_path = '../of500_images_resized/'

        ct, p1, p2 = 0, 0, 0
        # for batch_idx, batch_idx_ in zip(y, y_):
        #     batch = y[batch_idx]
        #     batch_ = y_[batch_idx_]
        batch_idx = 0
        print('visualize image:', image_id_batch)

        if y_verb_[0] == y_verb[0]:
            f_html = open(os.path.join(sr_visualpath, 'verb_correct', '%s.html' % (image_id_batch[0])), 'w')
        else:
            f_html = open(os.path.join(sr_visualpath, 'verb_wrong', '%s.html' % (image_id_batch[0])), 'w')
        f_html.write("<img src=\"" + os.path.join(image_path, image_id_batch[0]) + "\" width=\"300\">\n<br>")

        f_html.write('[verb_prediction] %s \n<br>' % verb_id2s[y_verb_[0]])
        f_html.write('[verb_ground_truth] %s \n<br>\n<br>' % verb_id2s[y_verb[0]])

        for batch, batch_ in zip(y, y_):
            # print('image_id', image_id_batch[batch_idx])
            p1 += len(batch)
            p2 += len(batch_)
            for role in batch:
                is_correct = False
                entities = batch[role]
                if role in batch_:
                    entities_ = batch_[role]
                    for entity_ in entities_:
                        if entity_ in entities:
                            ct += 1
                            is_correct = True
                            break
                    if not is_correct:
                        f_html.write('[prediction]\n<br>')
                        f_html.write('%s.%s = [' % (verb_id2s[y_verb_[batch_idx]], role_id2s[role]))
                        for entity_ in entities_:
                            f_html.write('%s, ' % noun_id2s[entity_])
                        f_html.write(']\n<br>')
                        f_html.write('[ground truth]\n<br>')
                        f_html.write('%s.%s = [' % (verb_id2s[y_verb[batch_idx]], role_id2s[role]))
                        for entity in entities:
                            f_html.write('%s, ' % noun_id2s[entity])
                        f_html.write(']\n\n<br><br>')
            batch_idx = batch_idx + 1
        f_html.flush()
        f_html.close()

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1