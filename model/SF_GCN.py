import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool, GATv2Conv, GCNConv
from typing import List, Tuple
import numpy as np
import torch.nn.init as init
import time
from functools import wraps
import math

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


class pred_Layer(nn.Module):
    def __init__(self
                , dim_per_layer: List
                , activation=nn.ELU()
                , dropout: float = 0.2) -> None:

        super(pred_Layer, self).__init__()
        
        layer_s = []
        for i in range(len(dim_per_layer) - 2):
            
            layer_s.append(nn.Linear(dim_per_layer[i], dim_per_layer[i + 1]))
            layer_s.append(activation)
            if i == len(dim_per_layer) - 2 -1:
                layer_s.append(nn.Dropout(dropout))
        
        layer_s.append(nn.Linear(dim_per_layer[-2], dim_per_layer[-1]))
        self.model = nn.Sequential(*layer_s)
        
        initialize_weights(self.model)

    def forward(self, x: torch.Tensor):
        return self.model(x)

class SF_GCN_Layers(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    n_GCN_hidden_features: GCN 隐藏层维度
    out_n_GCN_features: GCN 最后一层的输出
    n_GCN_layers: GATv2卷积层数
    dropout: 所有模型的dropout
    """

    def __init__(self,
                 in_n_node_features: int,
                 n_GCN_hidden_features: int,
                 out_n_GCN_features: int,
                 n_GCN_layers: int,
                 dropout: float):
        super(SF_GCN_Layers, self).__init__()

        self.convs = nn.ModuleList()
        in_channels = in_n_node_features

        for i in range(n_GCN_layers):
            out_channels = out_n_GCN_features if i == n_GCN_layers - 1 else n_GCN_hidden_features
            conv = GCNConv(in_channels=in_channels,
                           out_channels=out_channels)
            self.convs.append(conv)
            in_channels = out_channels

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)            # GCNConv 不需要 edge_attr
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        gnn_feat = torch.cat([x_max, x_mean], dim=1)
        return gnn_feat


class SF_GCN_MLP(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    n_GCN_hidden_features: CGN 隐藏层维度
    out_n_GCN_features: GCN 最后一层的输出
    n_GCN_layers: GATv2卷积层数
    dropout: 所有模型的dropout
    in_n_iaw_features: 额外增加的分子描述符
    n_pred_hidden_features: 输出头的隐藏层的维度
    n_pred_hidden_layers: 输出头隐藏层的数目
        - 还需要+2, 第一个是输入的变换, 最后一个是输出的变化
    merge_sign: 是否需要拼接额外的分子描述符
        - None: 不拼接
        - Cat: 拼接
    """

    def __init__(self
            , in_n_node_features: int  
            , n_GCN_hidden_features: int        
            , out_n_GCN_features: int
            , n_GCN_layers: int         
            , dropout: float
            , in_n_iaw_features: int
            , n_pred_hidden_features: int 
            , n_pred_hidden_layers: int 
            , merge_sign: str):             

        super(SF_GCN_MLP, self).__init__()

        self.merge_sign = merge_sign
        
        self.SF_GCN_Layers = SF_GCN_Layers(
              in_n_node_features
            , n_GCN_hidden_features
            , out_n_GCN_features
            , n_GCN_layers
            , dropout
        )
        if self.merge_sign == "Cat":
            
            # 对拼接的描述符先进行一次学习
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LeakyReLU(negative_slope=0.2)
                                    , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LayerNorm(in_n_iaw_features))

            # 拼接图的embedding以及拼接的MolDes
            self.merge_feature_dim = out_n_GCN_features*2+in_n_iaw_features
            self.merge_embed = nn.Sequential(nn.Linear(self.merge_feature_dim, self.merge_feature_dim)
                                                 , nn.LeakyReLU(negative_slope=0.2)
                                                 , nn.Linear(self.merge_feature_dim, self.merge_feature_dim))
        elif self.merge_sign == "None":
            self.merge_embed = nn.Identity()
            self.merge_feature_dim = out_n_GCN_features*2
        
        # 设置输出头
        self.pred_out = pred_Layer(dim_per_layer = [self.merge_feature_dim ] + [n_pred_hidden_features]*n_pred_hidden_layers +[1]
                                                        , dropout = dropout)

    def hidden_embedding(self , data):
        iaw_attr = data.iaw_attr
        gnn_feat = self.SF_GCN_Layers(data)
        
        # 这里需要做描述符的拼接
        if self.merge_sign == "Cat":
            iaw_feat = self.iaw_embed(iaw_attr)
            merge_feat = torch.cat([gnn_feat, iaw_feat], dim=1)
            #print("merge_feat", merge_feat.size(), "gnn_feat", gnn_feat.size(), "iaw_feat", iaw_feat.size())
            out_feat = self.merge_embed(merge_feat)
            #print("out_feat", out_feat.size())
        elif self.merge_sign == "None":
            out_feat = self.merge_embed(gnn_feat)
            
        return gnn_feat, out_feat

    def forward(self, data: Data):
        
        gnn_embedding, merge_embedding = self.hidden_embedding(data)

        out = self.pred_out(merge_embedding)

        return out


