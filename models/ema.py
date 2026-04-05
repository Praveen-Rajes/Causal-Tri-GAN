"""
CausalTriGAN-StyleGAN2 - Exponential Moving Average
Maintains a shadow copy of generator weights for stable evaluation.
"""
import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model, decay=0.9999, start_step=5000):
        self.decay = decay
        self.start_step = start_step
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model, step):
        if step < self.start_step:
            self._copy_params(model)
            return
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
            for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
                s_buf.data.copy_(m_buf.data)

    def _copy_params(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.copy_(m_param.data)
            for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
                s_buf.data.copy_(m_buf.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    class _EMAContext:
        def __init__(self, model, shadow):
            self.model = model
            self.shadow = shadow
            self.backup = {}

        def __enter__(self):
            self.backup = {name: p.data.clone() for name, p in self.model.named_parameters()}
            for (name, p), (_, s) in zip(self.model.named_parameters(),
                                          self.shadow.named_parameters()):
                p.data.copy_(s.data)
            return self.model

        def __exit__(self, *args):
            for name, p in self.model.named_parameters():
                p.data.copy_(self.backup[name])

    def average_parameters(self, model):
        return self._EMAContext(model, self.shadow)
