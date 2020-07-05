from src.models.modules.model import Model
from src.util.util_model import BottledXavierLinear
from torch import nn

class ACEClassifier(Model):
    def __init__(self, common_dim, type_num, role_num, device):
        # self.common_dim, hyps["oc"], hyps["ae_oc"]
        super(ACEClassifier, self).__init__()

        self.device = device

        # Output Linear
        self.ol = BottledXavierLinear(in_features=common_dim, out_features=type_num).to(device=device)

        # AE Output Linear
        self.ae_ol = BottledXavierLinear(in_features=2 * common_dim, out_features=role_num).to(device=device)
        # self.ae_l1 = nn.Linear(in_features=2 * common_dim, out_features=common_dim)
        # self.ae_bn1 = nn.BatchNorm1d(num_features=common_dim)
        # self.ae_l2 = nn.Linear(in_features=common_dim, out_features=role_num)

        # Move to right device
        self.to(self.device)

    def forward_type(self, feature_in):
        ed_logits = self.ol(feature_in)
        return ed_logits

    def forward_role(self, entity_feature_in):
        ae_logits = self.ae_ol(entity_feature_in)
        return ae_logits


