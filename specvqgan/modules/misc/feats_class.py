import torch

class FeatsClassStage(object):
    def __init__(self):
        pass

    def eval(self):
        return self

    def encode(self, c):
        """fake vqmodel interface because self.cond_stage_model should have something
        similar to coord.py but even more `dummy`"""
        # assert 0.0 <= c.min() and c.max() <= 1.0
        info = None, None, c
        return c, None, info

    def decode(self, c):
        return c

    def get_input(self, batch: dict, keys: dict) -> dict:
        out = {}
        for k in keys:
            if k == 'target':
                out[k] = batch[k].unsqueeze(1)
            elif k == 'feature':
                out[k] = batch[k].float().permute(0, 2, 1)
            out[k] = out[k].to(memory_format=torch.contiguous_format)
        return out
