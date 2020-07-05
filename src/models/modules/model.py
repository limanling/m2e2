import torch
import torch.nn as nn

import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util.util_model import log


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.hyperparams = None
        self.device = torch.device("cpu")

    def __getnewargs__(self):
        # for pickle
        return self.hyperparams

    def __new__(cls, *args, **kwargs):
        log('created %s with params %s' % (str(cls), str(args)))

        instance = super(Model, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    def test_mode_on(self):
        self.test_mode = True
        self.eval()

    def test_mode_off(self):
        self.test_mode = False
        self.train()

    def parameters_requires_grads(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def parameters_requires_grad_clipping(self):
        return self.parameters_requires_grads()

    def save_model(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            # print('save key', key)
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)

    def load_model(self, path, load_partial=False):
        pretrained_dict = torch.load(path)
        try:
            self.load_state_dict(pretrained_dict)
        except Exception as e:
            if load_partial:
                # load matched part
                model_dict = self.state_dict()
                ignore_keys = set(['ace_classifier.ol.linear.weight',
                                   'ace_classifier.ol.linear.bias',
                                   'ace_classifier.ae_ol.linear.weight',
                                   'ace_classifier.ae_ol.linear.bias'])
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ignore_keys} #in ignore_keys}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.load_state_dict(model_dict)
            else:
                print(e)
                exit(-1)
