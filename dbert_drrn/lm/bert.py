import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_utils import *
from jericho.util import clean
from jericho.defines import ILLEGAL_ACTIONS, NO_EFFECT_ACTIONS

from .base_lm import BaseLM, device


class BERTLM(BaseLM):
    def load_model(self, model_path):
        self.config = BertConfig.from_pretrained(model_path)
        self.config.is_decoder = True
        self.model = BertModel.from_pretrained(model_path, config=self.config)
        self.model.eval()
        self.model.to(device)

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})

    def act2ids(self, act):
        ret = self.tokenizer.encode(clean(act), add_prefix_space=True)
        if not ret: ret = [0]
        return ret

    def sent2ids(self, sent, maxlen=512):
        ret = self.tokenizer.encode(clean(sent))
        if len(ret) > maxlen:
            ret = ret[-maxlen:]
        if not ret: ret = [0]
        return ret

    def score(self, input, acts):
        input_ids = self.sent2ids(input) if isinstance(input, str) else input
        input_len = len(input_ids)
        input_ids = torch.tensor([input_ids]).to(device)
        scores = []
        for act in acts.copy():
            if isinstance(act, str):
                act = self.act2ids(act) + [50258]
            act_tensor = torch.tensor([act]).to(device)
            example = torch.cat((input_ids, act_tensor), axis=1)
            with torch.no_grad():
                predictions = self.model(example)[0][0][input_len - 1:-1]
            log_p = torch.nn.functional.log_softmax(predictions, dim=-1)
            scores.append(log_p[range(len(act)), act].sum().item())
        return scores