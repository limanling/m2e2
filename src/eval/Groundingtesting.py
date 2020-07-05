
class GroundingTester():
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
                entities = batch[role]
                if role in batch_:
                    entities_ = batch_[role]
                    for entity_ in entities_:
                        if entity_ in entities:
                            ct += 1
                            break

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

