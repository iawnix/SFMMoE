import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from typing import List, Tuple
import numpy as np
import torch.nn.init as init
import time
from functools import wraps
import math

# 定义一个装饰器来计算函数的运行时间
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  
        print(f"Func {func.__name__} run: {end_time - start_time} s")
        return result
    return wrapper

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

# 增加concat, 最后一次才合并多头与Gatv2保持一致
class KAFConv(nn.Module):
    def __init__(self
            , in_n_node_features: int
            , out_n_node_features: int
            , n_heads: int
            , in_n_edge_features: int
            , in_n_angle_features: int 
            , dropout: float
            , hop_2: bool
            , concat: bool
            ):
        
        super(KAFConv, self).__init__()

        self.in_n_node_features = in_n_node_features
        self.out_n_node_features = out_n_node_features
        self.n_heads = n_heads
        self.in_n_edge_features = in_n_edge_features
        self.in_n_angle_features = in_n_angle_features
        self.hop_2 = hop_2
        self.concat = concat

        # 定义网络
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.dropnet = nn.Dropout(dropout)  

        self.MLP_atten_update_node = nn.Sequential(
                  nn.Linear(self.in_n_node_features + self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
        
        self.MLP_atten_update_pos = nn.Sequential(
                  nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.out_n_node_features * self.n_heads, 1, bias=False))


        # hop1
        self.MLP_atten_edge = nn.Sequential(
                  nn.Linear(self.in_n_node_features*2 + self.in_n_edge_features+1, self.out_n_node_features * self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
        
        self.Linear_sum_edge = nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
        self.Linear_sum_edge_pos = nn.Linear(3, 3)
        self.a_edge = nn.Parameter(torch.Tensor(1, self.n_heads, self.out_n_node_features))
        
        # hop2
        if self.hop_2:
            self.MLP_atten_angle = nn.Sequential(
                  nn.Linear(self.in_n_node_features*3+self.in_n_edge_features*2+self.in_n_angle_features + 1, self.out_n_node_features*self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
            
            self.Linear_sum_angle = nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
            self.Linear_sum_angle_pos = nn.Linear(3, 3)
            self.a_angle = nn.Parameter(torch.Tensor(1, self.n_heads, self.out_n_node_features))  


        self.Linear_sum_x = nn.Linear(self.in_n_node_features, self.n_heads * self.out_n_node_features)
        self.Linear_sum_x_pos = nn.Linear(3, 3)
        

        # 残差层
        if self.in_n_node_features == self.n_heads * self.out_n_node_features:
            self.Linear_resnet = nn.Identity()
        else:
            self.Linear_resnet = nn.Linear(self.in_n_node_features, self.n_heads*self.out_n_node_features, bias=False)
        
        # bais and MHA concat
        if concat:                                            
            self.bias = nn.Parameter(torch.Tensor(self.out_n_node_features * self.n_heads))  
        else:                         
            self.bias = nn.Parameter(torch.Tensor(self.out_n_node_features))  
            

        # 最终输出的norm层
        if concat:
            self.layer_norm = nn.LayerNorm(self.out_n_node_features * self.n_heads)
        else:
            self.layer_norm = nn.LayerNorm(self.out_n_node_features)
        # 最终的激活函数
        self.activation = nn.PReLU()

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):

        initialize_weights(self.MLP_atten_update_node)
        initialize_weights(self.MLP_atten_update_pos)
        initialize_weights(self.MLP_atten_edge)
        nn.init.xavier_uniform_(self.Linear_sum_edge.weight)
        nn.init.xavier_uniform_(self.Linear_sum_x.weight)
        nn.init.xavier_uniform_(self.Linear_sum_edge_pos.weight)
        nn.init.xavier_uniform_(self.Linear_sum_x_pos.weight)
        nn.init.xavier_uniform_(self.a_edge)
    
        if self.hop_2:
            nn.init.xavier_uniform_(self.Linear_sum_angle.weight)
            initialize_weights(self.MLP_atten_angle)
            nn.init.xavier_uniform_(self.Linear_sum_angle_pos.weight)
            nn.init.xavier_uniform_(self.a_angle)
            
        if self.in_n_node_features != self.n_heads * self.out_n_node_features:
            nn.init.xavier_uniform_(self.Linear_resnet.weight)
        
        nn.init.constant_(self.bias, 0)
    
    def index_edge_i_j(self, edge_index, i, j):
        row = torch.where((edge_index[0,:] == i))[0]
        col = torch.where((edge_index[1,:] == j))[0]
        return row[torch.isin(row,col)]
    
    @timing_decorator
    def index_edge_is_js(self, edge_index, i_s, j_s):
        out = []
        for _, i in enumerate(i_s):
            out.append(self.index_edge_i_j(edge_index, i, j_s[_]))
        return torch.cat(out, dim=0)

    def atten_update_node(self, x, _a_, _a_project, j_):
        e_ij_x_i_result_shape = (x.size()[0], self.n_heads, self.out_n_node_features)
        e_ij_x_i_result = _a_.new_full(e_ij_x_i_result_shape, 0)
        e_ij_x_i_broadcast = (j_.unsqueeze(-1)).unsqueeze(-1).expand_as(_a_project)
        e_ij_x_i_result.scatter_add_(0, e_ij_x_i_broadcast, _a_project)
        e_ij_x_i_result = e_ij_x_i_result.view(x.size()[0], self.n_heads * self.out_n_node_features)
        e_ij_x_i_agg = torch.cat([x, e_ij_x_i_result], dim = 1)

        update_node = self.MLP_atten_update_node(e_ij_x_i_agg).view(-1, self.n_heads, self.out_n_node_features)
        return update_node

    # 相当于加入e3等变
    # 修订加入C, 解决稀疏图
    # https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L106
    def atten_update_pos(self, pos, _a_, _a_project, j_, _diff, _radial):
        trans = _diff * self.MLP_atten_update_pos(_a_project.view(-1, self.out_n_node_features * self.n_heads))
        coord_result_shape = (pos.size(0), trans.size(1))
        coord_result = trans.new_full(coord_result_shape, 0)
        count = trans.new_full(coord_result_shape, 0)
        dst_index_broadcast = j_.unsqueeze(-1).expand(-1, trans.size(1))
        coord_result.scatter_add_(0, dst_index_broadcast, trans)
        count.scatter_add_(0, dst_index_broadcast, torch.ones_like(trans))
        update_pos = pos + coord_result/count.clamp(min=1)
        return update_pos
    
    def atten_edge(self, x, edge_index, edge_attr, pos):
        #################################################################
        ## A <----- B
        ## 其实是考察的B对A的影响

        #################################################################
        ## edge的多头注意力
        ##          e_ij = leakReLU(a * cat([x_i, x_j, e_ij_f]))
        ##          α_ij = softmax(e_ij) = exp(e_ij) / sum(exp(e_ik))
        ## BEGIN: e_ij及exp(e_ij)求解
        i_ = edge_index[0,:]
        j_ = edge_index[1,:]
        edge_coord_diff = pos[j_] - pos[i_]   
        edge_radial = torch.sum(edge_coord_diff**2, 1).unsqueeze(1)  

        # 归一化，否则会出现nan
        norm = torch.sqrt(edge_radial).detach() + 1e-16  
        edge_coord_diff = edge_coord_diff / norm 

        e_ij_f = torch.cat( [x[i_]
                         , x[j_]
                         , edge_attr
                         , norm], dim=1)                                                                                                               # 这俩是一对一的关系
        e_ij_f = self.MLP_atten_edge(e_ij_f).view(-1, self.n_heads, self.out_n_node_features)                   # 这里需要一个线性层
        # Gat V2
        e_ij = self.leakyReLU(e_ij_f)
        e_ij = (self.a_edge * e_ij).sum(-1)
        ## END: e_ij

        e_ij_exp = (e_ij - e_ij.max()).exp()
        ## END: exp(ee_ij)

        ## BEGIN: sum(exp(e_ik))及α_ij求解
        e_ij_exp_sum_shape = (x.size()[0], self.n_heads)
        e_ij_exp_sum = e_ij_exp.new_full(e_ij_exp_sum_shape, 0)
        e_j_broadcast = j_.unsqueeze(-1).expand_as(e_ij_exp)
        e_ij_exp_sum.scatter_add_(0, e_j_broadcast, e_ij_exp)
        _a_ij = e_ij_exp / (e_ij_exp_sum.index_select(0, j_)+ 1e-16)
        _a_ij = self.dropnet(_a_ij).unsqueeze(-1)
        ## END

        _a_ij_project = e_ij_f * _a_ij

        return _a_ij, _a_ij_project, j_, edge_coord_diff, norm

    def atten_angle(self, x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos):
        #################################################################
        ## A <----- B
        ##               ^
        ##                 \
        ##                  \
        ##                   C
        ## 其实是考察的C对A的影响

        ## BEGIN: e_ij及exp(e_ij)求解
        i_ = angle_index[0,:]
        h_ = angle_index[1,:]
        j_ = angle_index[2,:]
        angle_coord_diff = (pos[h_] - pos[i_]  ) + (pos[j_] - pos[h_] )
        angle_radial = torch.sum(angle_coord_diff**2, 1).unsqueeze(1)   
        # 归一化
        norm = torch.sqrt(angle_radial).detach() + 1e-16  
        angle_coord_diff = angle_coord_diff / norm 
        
        # 这里是否要区分有向跟无向
        angle_ihj_f = torch.cat( [x[i_]
                              , x[h_]
                              , x[j_]
                              , angle_edge_attr[:,0]
                              , angle_edge_attr[:,1]
                              , angle_attr* (math.pi / 180)
                              , norm], dim=1)
        angle_ihj_f = self.MLP_atten_angle(angle_ihj_f).view(-1, self.n_heads, self.out_n_node_features)            # 这里需要一个线性层
        # Gat V2
        angle_ihj = self.leakyReLU(angle_ihj_f)
        angle_ihj = (self.a_angle * angle_ihj).sum(-1)
        ## END: e_ij
        angle_ihj_exp = (angle_ihj - angle_ihj.max()).exp()
        ## END: exp(ee_ij)
        ## BEGIN: sum(exp(e_ik))及α_ij求解
        angle_ihj_exp_sum_shape = (x.size()[0], self.n_heads)
        angle_ihj_exp_sum = angle_ihj_exp.new_full(angle_ihj_exp_sum_shape, 0)
        angle_ihj_broadcast = j_.unsqueeze(-1).expand_as(angle_ihj_exp)
        angle_ihj_exp_sum.scatter_add_(0, angle_ihj_broadcast, angle_ihj_exp)
        _a_ihj = angle_ihj_exp / (angle_ihj_exp_sum.index_select(0, j_)+ 1e-16)
        _a_ihj = self.dropnet(_a_ihj).unsqueeze(-1)

        _a_ihj_project = angle_ihj_f * _a_ihj
        return _a_ihj, _a_ihj_project, j_, angle_coord_diff, norm
    
    def forward(self, data: List):
        
        # 最后一位是squence_attn的平均
        x, edge_index , edge_attr , angle_index, angle_attr, angle_edge_attr, pos, _ = data

        # message
        # hop1
        edge_a_, edge_a_project, dst_,  edge_coord_diff, edge_radial = self.atten_edge(x, edge_index, edge_attr, pos)
        x_edge_out = self.atten_update_node(x, edge_a_, edge_a_project, dst_)
        pos_edge_out = self.atten_update_pos(pos, edge_a_, edge_a_project, dst_, edge_coord_diff, edge_radial)

        # hop2
        if self.hop_2:
            angle_a_, angle_a_project, dst_, angle_coord_diff, angle_radial = self.atten_angle(x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos)
            x_angle_out = self.atten_update_node(x, angle_a_, angle_a_project, dst_)
            pos_angle_out = self.atten_update_pos(pos, angle_a_, angle_a_project, dst_, angle_coord_diff, angle_radial)
        
        # message sum
        x_edge_out = self.Linear_sum_edge(x_edge_out.view(-1, self.n_heads*self.out_n_node_features)).view(-1, self.n_heads, self.out_n_node_features)

        if self.hop_2:

            # 2-hop注意力计算，分别计算α_ab, α_abc与a之间的注意力分数,类似QKV
            x_angle_out = self.Linear_sum_angle(x_angle_out.view(-1, self.n_heads*self.out_n_node_features)).view(-1, self.n_heads, self.out_n_node_features)     
            x_ = self.Linear_sum_x(x).view(-1, self.n_heads, self.out_n_node_features)           

            edge_attn = torch.diagonal(torch.matmul(x_, x_edge_out.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.out_n_node_features)
            angle_attn = torch.diagonal(torch.matmul(x_, x_angle_out.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.out_n_node_features)
            squence_attn = torch.stack([edge_attn, angle_attn], dim=0)

            squence_attn = self.dropnet(torch.softmax(squence_attn, dim=0).unsqueeze(-1))
            
            # 2-hop最终的x_out以及pos_out
            x_out = squence_attn[0, :, :] * x_edge_out + squence_attn[1, :, :] * x_angle_out
            pos_out = pos_edge_out + pos_angle_out
        
        else:    
            # 1-hop最终的x_out以及pos_out
            x_out = x_edge_out
            pos_out = pos_edge_out


        x_out += self.Linear_resnet(x).view(-1, self.n_heads,self.out_n_node_features)  
        if self.concat:
            x_out = x_out.view(-1, self.n_heads * self.out_n_node_features)
        else:
            x_out = x_out.mean(dim = 1)

        x_out += self.bias

        x_out = self.layer_norm(x_out)                                                                                    # 这里需要一个归一化 
        x_out = self.activation(x_out)                                                                                      # 这里需要一个归一化层

        if self.hop_2:
            squence_attn_mean_list = [squence_attn.mean(dim = 2)[0,:,:], squence_attn.mean(dim = 2)[1,:,:]]
        else:    
            squence_attn_mean_list = []

        return [x_out, edge_index , edge_attr , angle_index, angle_attr, angle_edge_attr, pos_out, squence_attn_mean_list]


class SF_KAF_Layers(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    in_n_edge_features: edge的初始输入维度
    in_n_angle_features: angle的初始输入维度
    n_KAF_hidden_features: KAF隐藏层的维度
    out_n_KAF_features: KAF输出层的维度
    n_KAF_layers: KAF的层数
    n_KAF_heads: KAF的头数
    hop_2: 是否采用2hop的消息传递
    dropout: 所有模块的dropout

    """
    def __init__(self, 
                   in_n_node_features: int
                 , in_n_edge_features: int
                 , in_n_angle_features: int
                 , n_KAF_hidden_features: int
                 , out_n_KAF_features: int
                 , n_KAF_layers: int
                 , n_KAF_heads: int
                 , hop_2: bool
                 , concat_hidden : bool
                 , dropout: float):

        super(SF_KAF_Layers, self).__init__()
        self.hop_2 = hop_2

        if concat_hidden == True:

            convs = []
            in_channels = in_n_node_features
            for i in range(n_KAF_layers):

                # 没有到最后一层的时候的都是concat
                i_concat = i < n_KAF_layers - 1
                # 最后一层的时候等于out_n_KAF_features, 其它时候为隐藏层
                out_channels = out_n_KAF_features if i == n_KAF_layers - 1 else n_KAF_hidden_features

                conv = KAFConv(in_n_node_features = in_channels
                                            , out_n_node_features = out_channels
                                            , n_heads = n_KAF_heads
                                            , in_n_edge_features = in_n_edge_features
                                            , in_n_angle_features = in_n_angle_features
                                            , dropout = dropout
                                            , concat = i_concat
                                            , hop_2 = hop_2)

                convs.append(conv)
                in_channels = out_channels * n_KAF_heads if i_concat else out_channels
        else:
            convs = []
            in_channels = in_n_node_features
            i_concat = False
            for i in range(n_KAF_layers):

                # 最后一层的时候等于out_n_KAF_features, 其它时候为隐藏层
                out_channels = out_n_KAF_features if i == n_KAF_layers - 1 else n_KAF_hidden_features

                conv = KAFConv(in_n_node_features = in_channels
                                            , out_n_node_features = out_channels
                                            , n_heads = n_KAF_heads
                                            , in_n_edge_features = in_n_edge_features
                                            , in_n_angle_features = in_n_angle_features
                                            , dropout = dropout
                                            , concat = i_concat
                                            , hop_2 = hop_2)

                convs.append(conv)
                in_channels =  out_channels

        # gnn_embed
        self.gnn_embed = nn.Sequential(nn.Linear(out_n_KAF_features*2, out_n_KAF_features*2)
                                     , nn.LeakyReLU(negative_slope=0.2)
                                     , nn.Linear(out_n_KAF_features*2, out_n_KAF_features*2)
                                     , nn.LayerNorm(out_n_KAF_features*2))

        self.convs = nn.Sequential(*convs)

    def forward(self, data):  
        x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.angle_index, data.angle_attr, data.angle_edge_attr, data.pos
        for conv in self.convs:
            x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos, k_hop_atten_mean = conv([x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos, []])

        # last 
        x_out = x

        if self.hop_2:
            hop_1_atten, hop_2_atten = k_hop_atten_mean
            if data.batch == None:
                hop_1_atten = [hop_1_atten]
                hop_2_atten = [hop_2_atten]
            else:
                split_sizes = torch.bincount(data.batch).tolist()
                hop_1_atten = torch.split(hop_1_atten, split_sizes)
                hop_2_atten = torch.split(hop_2_atten, split_sizes)
            
        else:
            hop_1_atten = []
            hop_2_atten = []
        # 池化
        output1 = global_max_pool(x_out, data.batch)
        output2 = global_mean_pool(x_out , data.batch)
        gnn_embed = self.gnn_embed(torch.cat([output1, output2], dim=1))

 
        return gnn_embed, [hop_1_atten, hop_2_atten]

class SF_KAF_MLP(nn.Module):
    """
    in_n_node_features: node的初始输入维度
    in_n_edge_features: edge的初始输入维度
    in_n_angle_features: angle的初始输入维度
    n_KAF_hidden_features: KAF隐藏层的维度
    out_n_KAF_features: KAF输出层的维度
    n_KAF_layers: KAF的层数
    n_KAF_heads: KAF的头数
    hop_2: 是否采用2hop的消息传递
    concat_hidden: 隐藏层的多头是否采用concat拼接
    dropout: 所有模块的dropout
    in_n_iaw_features: 额外增加的分子描述符
    n_pred_hidden_features: 输出头的隐藏层的维度
    n_pred_hidden_layers: 输出头隐藏层的数目
        - 还需要+2, 第一个是输入的变换, 最后一个是输出的变化
    merge_sign: 是否需要拼接额外的分子描述符
        - None: 不拼接
        - Cat: 拼接
    """
    def __init__(self,in_n_node_features: int
                 , in_n_edge_features: int
                 , in_n_angle_features: int
                 , n_KAF_hidden_features: int
                 , out_n_KAF_features: int
                 , n_KAF_layers: int
                 , n_KAF_heads: int
                 , hop_2: bool
                 , concat_hidden: bool
                 , dropout: float
                 , in_n_iaw_features: int
                 , n_pred_hidden_features: int
                 , n_pred_hidden_layers: int
                 , merge_sign: str):

        super(SF_KAF_MLP, self).__init__()

        self.merge_sign = merge_sign

        self.SF_KAF_Layers = SF_KAF_Layers(in_n_node_features
                 , in_n_edge_features
                 , in_n_angle_features
                 , n_KAF_hidden_features
                 , out_n_KAF_features
                 , n_KAF_layers
                 , n_KAF_heads
                 , hop_2
                 , concat_hidden
                 , dropout)

        if self.merge_sign == "Cat":
            
            # 对拼接的描述符先进行一次学习
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LeakyReLU(negative_slope=0.2)
                                    , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                    , nn.LayerNorm(in_n_iaw_features))

            # 拼接图的embedding以及拼接的MolDes
            self.merge_feature_dim = out_n_KAF_features*2+in_n_iaw_features
            self.merge_embed = nn.Sequential(nn.Linear(self.merge_feature_dim, self.merge_feature_dim)
                                                 , nn.LeakyReLU(negative_slope=0.2)
                                                 , nn.Linear(self.merge_feature_dim, self.merge_feature_dim))
        elif self.merge_sign == "None":
            self.merge_embed = nn.Identity()
            self.merge_feature_dim = out_n_KAF_features*2
 
        # 设置输出头
        self.pred_out = pred_Layer(dim_per_layer = [self.merge_feature_dim ] + [n_pred_hidden_features]*n_pred_hidden_layers +[1]
                                                        , dropout = dropout)



    def hidden_embedding(self, data):
        gnn_embed, k_hop_atten_mean = self.SF_KAF_Layers(data)
        iaw_attr = data.iaw_attr

        # 定义一个变量用于输出, 只是为了写一个return
        hop_1_atten, hop_2_atten = k_hop_atten_mean
        attention_matrix = None

        # 这里需要做描述符的拼接
        if self.merge_sign == "Cat":
            iaw_feat = self.iaw_embed(iaw_attr)
            merge_feat = torch.cat([gnn_embed, iaw_feat], dim=1)
            out_feat = self.merge_embed(merge_feat)
        elif self.merge_sign == "None":
            out_feat = self.merge_embed(gnn_embed)
            
        return out_feat, [hop_1_atten, hop_2_atten, attention_matrix]

    def forward(self, data):
        merge_embedding, attens = self.hidden_embedding(data)
        out = self.pred_out(merge_embedding)

        return out

