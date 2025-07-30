import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """一个带有两个全连接层的残差块"""
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        out += residual  # 残差连接
        return F.relu(out)

class ResNetMLP(nn.Module):
    """一个使用残差块构建的MLP"""
    def __init__(self, input_dim, output_dim, hidden_size=256, num_blocks=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x) 