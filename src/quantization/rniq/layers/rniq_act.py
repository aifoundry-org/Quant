from torch import nn


class NoisyAct(nn.Module):
    def __init__(self, init_s=-10, init_q=10, signed=True, rand_noise=False) -> None:
        super().__init__()
