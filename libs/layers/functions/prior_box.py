import numpy as np
import math
import torch
import itertools


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.input_size = cfg['input_shape']
        self.sizes = cfg['sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.output_stride = cfg['output_stride']
        self.clip = cfg['clip']
        self.variances = cfg['variances']
        self.with_max_size = (True, False)[len(self.sizes) == len(self.output_stride)]

        self.steps_w = [s / self.input_size[0] for s in self.output_stride]
        self.steps_h = [s / self.input_size[1] for s in self.output_stride]
        self.fm_sizes = [[int(np.ceil(1.0 / step_h)), int(np.ceil(1.0 / step_w))]
                             for step_w, step_h in zip(self.steps_w, self.steps_h)]

    def forward(self):
        priors = []
        for i, fm_size in enumerate(self.fm_sizes):
            fm_h, fm_w = fm_size
            for h, w in itertools.product(range(fm_h), range(fm_w)):
                cx = (w + 0.5) * self.steps_w[i]
                cy = (h + 0.5) * self.steps_h[i]

                s = self.sizes[i]
                priors.append((cx, cy, s, s))

                if self.with_max_size:
                    s = math.sqrt(self.sizes[i] * self.sizes[i+1])
                    priors.append((cx, cy, s, s))
                    s = self.sizes[i]

                for ar in self.aspect_ratios[i]:
                    priors.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    priors.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        priors = torch.Tensor(priors) # xywh
        if self.clip:
            priors.clamp_(min=0., max=1.)
        priors = priors.to('cuda')

        return priors
