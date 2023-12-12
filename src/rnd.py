from torch import nn
from torch.nn import init
import numpy as np


class RNDModel(nn.Module):
    # def __init__(self, input_size):
    #     super(RNDModel, self).__init__()
    def __init__(self, list_of_input_sizes):
        super(RNDModel, self).__init__()
        for i, input_size in enumerate(list_of_input_sizes):
            setattr(self, f"task_{i}", nn.Linear(input_size, 32))

        # self.output_size = output_size
        self.predictor = nn.Sequential(
            #nn.Linear(self.input_size[0], 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_size[0])
        )

        self.target = nn.Sequential(
            #nn.Linear(self.input_size[0], 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_size[0])
        )
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature