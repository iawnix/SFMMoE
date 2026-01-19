import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool, GATv2Conv
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

class SF_Gatv2_Layers(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    in_n_edge_features: edge的初始输入维度 
    n_Gatv2_hidden_features: Gatv2 隐藏层维度
    out_n_Gatv2_features: GatV2最后一层的输出
    n_Gatv2_layers: GATv2卷积层数
    n_Gatv2_heads: Gatv2每层的注意力头数
    dropout: 所有模型的dropout
    """

    def __init__(self
            , in_n_node_features: int  
            , in_n_edge_features: int   
            , n_Gatv2_hidden_features: int          
            , out_n_Gatv2_features: int
            , n_Gatv2_layers: int          
            , n_Gatv2_heads: int  
            , dropout: float):             

        super(SF_Gatv2_Layers, self).__init__()

        self.in_n_edge_features = in_n_edge_features
        # 设置卷积层
        self.convs = nn.ModuleList()
        in_channels = in_n_node_features
        
        for i in range(n_Gatv2_layers):
      
            #i_head = n_Gatv2_heads if i < n_Gatv2_layers - 1 else 1
            # 之前相当于少了一层Conv
            i_head = n_Gatv2_heads 
            # 最后一层的时候要求平均, 需要设置为Flase
            # True: out = n_head * out_dim
            # Flase: out = out_dim
            i_concat = i < n_Gatv2_layers - 1
            
            out_channels = out_n_Gatv2_features if i == (n_Gatv2_layers - 1)  else n_Gatv2_hidden_features
            
            conv = GATv2Conv(
                  in_channels = in_channels
                , out_channels = out_channels
                , edge_dim = in_n_edge_features
                , heads = i_head
                , concat = i_concat
                , dropout = dropout)

            self.convs.append(conv)
            # 计算下一次Gatv2卷积的维度
            in_channels = out_channels * i_head if i_concat else out_channels
            
            
    def forward(self, data: Data):
        if self.in_n_edge_features != 0:
            x, edge_index, edge_attr, iaw_attr, batch = data.x, data.edge_index, data.edge_attr, data.iaw_attr, data.batch

            # GATv2 消息传递
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                #print("x", x.size())
            # 全局池化
            x_max = global_max_pool(x, batch)  
            x_mean = global_mean_pool(x, batch)   
            gnn_feat = torch.cat([x_max, x_mean], dim=1)
        
        else:
            # 不传入边特征
            x, edge_index, edge_attr, iaw_attr, batch = data.x, data.edge_index, data.edge_attr, data.iaw_attr, data.batch

            # GATv2 消息传递
            for conv in self.convs:
                x = conv(x, edge_index)
                #print("x", x.size())
            # 全局池化
            x_max = global_max_pool(x, batch)  
            x_mean = global_mean_pool(x, batch)   
            gnn_feat = torch.cat([x_max, x_mean], dim=1)

        return gnn_feat


class SF_Gatv2_MLP(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    in_n_edge_features: edge的初始输入维度 
    n_Gatv2_hidden_features: Gatv2 隐藏层维度
    out_n_Gatv2_features: GatV2最后一层的输出
    n_Gatv2_layers: GATv2卷积层数
    n_Gatv2_heads: Gatv2每层的注意力头数
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
            , in_n_edge_features: int
            , n_Gatv2_hidden_features: int        
            , out_n_Gatv2_features: int
            , n_Gatv2_layers: int         
            , n_Gatv2_heads: int   
            , dropout: float
            , in_n_iaw_features: int
            , n_pred_hidden_features: int 
            , n_pred_hidden_layers: int 
            , merge_sign: str):             

        super(SF_Gatv2_MLP, self).__init__()

        self.merge_sign = merge_sign
        
        self.SF_Gatv2_Layers = SF_Gatv2_Layers(
              in_n_node_features
            , in_n_edge_features
            , n_Gatv2_hidden_features
            , out_n_Gatv2_features
            , n_Gatv2_layers
            , n_Gatv2_heads
            , dropout
        )
        if self.merge_sign == "Cat":
            
            # 对拼接的描述符先进行一次学习
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LeakyReLU(negative_slope=0.2)
                                    , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LayerNorm(in_n_iaw_features))

            # 拼接图的embedding以及拼接的MolDes
            self.merge_feature_dim = out_n_Gatv2_features*2+in_n_iaw_features
            self.merge_embed = nn.Sequential(nn.Linear(self.merge_feature_dim, self.merge_feature_dim)
                                                 , nn.LeakyReLU(negative_slope=0.2)
                                                 , nn.Linear(self.merge_feature_dim, self.merge_feature_dim))
        elif self.merge_sign == "None":
            self.merge_embed = nn.Identity()
            self.merge_feature_dim = out_n_Gatv2_features*2
        
        # 设置输出头
        self.pred_out = pred_Layer(dim_per_layer = [self.merge_feature_dim ] + [n_pred_hidden_features]*n_pred_hidden_layers +[1]
                                                        , dropout = dropout)

    def hidden_embedding(self , data):
        iaw_attr = data.iaw_attr
        gnn_feat = self.SF_Gatv2_Layers(data)
        
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


