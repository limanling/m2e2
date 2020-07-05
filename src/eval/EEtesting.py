from seqeval.metrics import f1_score, precision_score, recall_score


class EDTester():
    def __init__(self, type_i2s, role_i2s, ignore_time):
        self.voc_i2s = type_i2s
        self.role_i2s = role_i2s
        self.ignore_time = ignore_time

    def calculate_report(self, y, y_, transform=True):
        '''
        calculating F1, P, R

        :param y: golden label, list
        :param y_: model output, list
        :return:
        '''
        if transform:
            for i in range(len(y)):
                for j in range(len(y[i])):
                    y[i][j] = self.voc_i2s[y[i][j]]
            for i in range(len(y_)):
                for j in range(len(y_[i])):
                    y_[i][j] = self.voc_i2s[y_[i][j]]
        return precision_score(y, y_), recall_score(y, y_), f1_score(y, y_)

    @staticmethod
    def merge_segments(y, send_id=None):
        segs = {}
        tt = ""
        st, ed = -1, -1
        for i, x in enumerate(y):
            if x.startswith("B-"):
                if tt == "":
                    tt = x[2:]
                    if send_id is None:
                        st = i
                    else:
                        st = '%s__%d' % (send_id, i)
                else:
                    ed = i
                    segs[st] = (ed, tt)
                    tt = x[2:]
                    if send_id is None:
                        st = i
                    else:
                        st = '%s__%d' % (send_id, i)
            elif x.startswith("I-"):
                if tt == "":
                    y[i] = "B" + y[i][1:]
                    tt = x[2:]
                    if send_id is None:
                        st = i
                    else:
                        st = '%s__%d' % (send_id, i)
                else:
                    if tt != x[2:]:
                        ed = i
                        segs[st] = (ed, tt)
                        y[i] = "B" + y[i][1:]
                        tt = x[2:]
                        if send_id is None:
                            st = i
                        else:
                            st = '%s__%d' % (send_id, i)
            else:
                ed = i
                if tt != "":
                    segs[st] = (ed, tt)
                tt = ""

        if tt != "":
            segs[st] = (len(y), tt)
        return segs

    def calculate_sets(self, y, y_):
        ct, p1, p2 = 0, 0, 0
        for sent, sent_ in zip(y, y_):
            # trigger start
            for key, value in sent.items():
                # key = trigger end, event type
                # value = args
                p1 += len(value)
                if key not in sent_:
                    continue
                # matched sentences
                arguments = value
                arguments_ = sent_[key]
                for item, item_ in zip(arguments, arguments_):
                    # print('item', self.role_i2s[item[2]], self.role_i2s[item_[2]])
                    if self.ignore_time and self.role_i2s[item[2]].upper().startswith('TIME'):
                        continue
                    if item[2] == item_[2]:
                        ct += 1

            for key, value in sent_.items():
                # p2_key += 1
                # p2 += len(value)
                # print('key', key)
                for item in sent_[key]:
                    if self.ignore_time and self.role_i2s[item[2]].upper().startswith('TIME'):
                        continue
                    p2 += 1


        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            print('ct', ct)
            print('p1', p1)
            print('p2', p2)
            return p, r, f1
