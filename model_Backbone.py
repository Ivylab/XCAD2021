import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    def __init__(self, classCount, drop_rate):
        super(DenseNet121, self).__init__()

        self.densenet121 = models.densenet121(pretrained=True, drop_rate=drop_rate)
        self.densenet121.classifier = Identity()
        self.l_logits = nn.Linear(1024, classCount)


    def forward(self, x):
        x = self.densenet121(x) # 32 * 1024
        logits = self.l_logits(x) #14

        return logits