from torch import nn

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FC(nn.Module):
    """Applies a linear transformation to the incoming data
    optionally followed by batch normalization and relu activation
    Attributes:
        fc (nn.Module): linear module
        bn (nn.Module): batch normalization module
        relu (nn.Module, optional): relu activation module
    """

    def __init__(self, in_channels, out_channels,
                 relu=True, bn=True, bn_momentum=0.1):
        super(FC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(in_channels, out_channels, bias=(not bn))
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x