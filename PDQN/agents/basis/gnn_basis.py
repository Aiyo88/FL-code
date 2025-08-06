import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GNNModel(nn.Module):
    """
    一个基于图注意力网络（GAT）的图神经网络模型。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_gat_layers=2):
        """
        Args:
            input_dim (int): 输入节点特征的维度。
            hidden_dim (int): GAT层的隐藏维度。
            output_dim (int): 最终输出的维度。
            num_heads (int): GAT中的多头注意力头数。
            num_gat_layers (int): GAT层的数量。
        """
        super(GNNModel, self).__init__()
        self.gat_layers = nn.ModuleList()
        
        # 输入层
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        )
        
        # 隐藏层
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
            )
            
        # 输出层 - 将图的表示映射到最终输出
        self.output_layer = nn.Linear(hidden_dim * num_heads, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): 包含 x, edge_index, batch 的图数据对象。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 通过GAT层
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x) # 使用ELU激活函数
        
        # 全局池化：从节点表示生成整个图的表示
        graph_embedding = global_mean_pool(x, batch)
        
        # 通过输出层
        output = self.output_layer(graph_embedding)
        
        return output 