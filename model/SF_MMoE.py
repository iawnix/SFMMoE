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
        start_time = time.time()  # 开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        print(f"函数 {func.__name__} 运行时间：{end_time - start_time} 秒")
        return result
    return wrapper

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
 

class KAF_Layer(nn.Module):
    def __init__(self
            , in_n_node_features: int
            , out_n_node_features: int
            , n_heads: int
            , in_n_edge_features: int
            , in_n_angle_features: int 
            ):
        
        super(KAF_Layer, self).__init__()

        self.in_n_node_features = in_n_node_features
        self.out_n_node_features = out_n_node_features
        self.n_heads = n_heads
        self.in_n_edge_features = in_n_edge_features
        self.in_n_angle_features = in_n_angle_features
        # 定义网络
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.dropnet = nn.Dropout(0.0)  

        self.MLP_atten_update_node = nn.Sequential(
                  nn.Linear(self.in_n_node_features + self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
        self.MLP_atten_update_pos = nn.Sequential(
                  nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.out_n_node_features * self.n_heads, 1, bias=False))

        self.MLP_atten_edge = nn.Sequential(
                  nn.Linear(self.in_n_node_features*2 + self.in_n_edge_features+1, self.out_n_node_features * self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
        self.MLP_atten_angle = nn.Sequential(
                  nn.Linear(self.in_n_node_features*3+self.in_n_edge_features*2+self.in_n_angle_features + 1, self.out_n_node_features*self.n_heads)
                #  nn.Linear(self.in_n_node_features*3+self.in_n_edge_features*2, self.out_n_node_features*self.n_heads)
                , nn.SiLU()
                , nn.Linear(self.n_heads * self.out_n_node_features, self.n_heads * self.out_n_node_features)
                )
        
        self.Linear_sum_edge = nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
        self.Linear_sum_angle = nn.Linear(self.out_n_node_features * self.n_heads, self.out_n_node_features * self.n_heads)
        self.Linear_sum_x = nn.Linear(self.in_n_node_features, self.n_heads * self.out_n_node_features)

        # 这里定义用与pos融合的注意力层
        self.Linear_sum_edge_pos = nn.Linear(3, 3)
        self.Linear_sum_angle_pos = nn.Linear(3, 3)
        self.Linear_sum_x_pos = nn.Linear(3, 3)

        # 定义注意力
        self.a_edge = nn.Parameter(torch.Tensor(1, self.n_heads, self.out_n_node_features))
        self.a_angle = nn.Parameter(torch.Tensor(1, self.n_heads, self.out_n_node_features))  

        # 残差层
        if self.in_n_node_features == self.n_heads * self.out_n_node_features:
            self.Linear_resnet = nn.Identity()
        else:
            self.Linear_resnet = nn.Linear(self.in_n_node_features, self.n_heads*self.out_n_node_features, bias=False)
        # bais
        self.bias = nn.Parameter(torch.Tensor(self.out_n_node_features))  

        # 最终输出的norm层
        self.layer_norm = nn.LayerNorm(self.out_n_node_features)

        # 最终的激活函数
        self.activation = nn.PReLU()

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        initialize_weights(self.MLP_atten_update_node)
        initialize_weights(self.MLP_atten_update_pos)
        initialize_weights(self.MLP_atten_edge)
        initialize_weights(self.MLP_atten_angle)

        nn.init.xavier_uniform_(self.Linear_sum_edge.weight)
        nn.init.xavier_uniform_(self.Linear_sum_angle.weight)
        nn.init.xavier_uniform_(self.Linear_sum_x.weight)

        nn.init.xavier_uniform_(self.Linear_sum_edge_pos.weight)
        nn.init.xavier_uniform_(self.Linear_sum_angle_pos.weight)
        nn.init.xavier_uniform_(self.Linear_sum_x_pos.weight)

        nn.init.xavier_uniform_(self.a_edge)
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
        #print(x.size(), e_ij_x_i_result.size())
        e_ij_x_i_result = e_ij_x_i_result.view(x.size()[0], self.n_heads * self.out_n_node_features)
        #print(x.size(), e_ij_x_i_result.size())
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
        edge_radial = torch.sqrt(torch.sum(edge_coord_diff**2, 1)).unsqueeze(1)  

        # 归一化，否则会出现nan
        norm = torch.sqrt(edge_radial).detach() + 1e-16  
        edge_coord_diff = edge_coord_diff / norm 

        e_ij_f = torch.cat( [x[i_]
                         , x[j_]
                         , edge_attr
                         , edge_radial], dim=1)           # 这俩是一对一的关系
        e_ij_f = self.MLP_atten_edge(e_ij_f).view(-1, self.n_heads, self.out_n_node_features)                   # 这里需要一个线性层
        # Gat V2
        e_ij = self.leakyReLU(e_ij_f)
        e_ij = (self.a_edge * e_ij).sum(-1)
        #e_ij = (self.a_edge * e_ij_f).sum(-1)
        #e_ij = self.leakyReLU(e_ij)
        ## END: e_ij
        e_ij_exp = (e_ij - e_ij.max()).exp()
        ## END: exp(ee_ij)
        ## BEGIN: sum(exp(e_ik))及α_ij求解
        # 计算一下花费的时间
        #print(e_ij_exp, j_)
        #start_time = time.time()
        e_ij_exp_sum_shape = (x.size()[0], self.n_heads)
        e_ij_exp_sum = e_ij_exp.new_full(e_ij_exp_sum_shape, 0)
        e_j_broadcast = j_.unsqueeze(-1).expand_as(e_ij_exp)
        e_ij_exp_sum.scatter_add_(0, e_j_broadcast, e_ij_exp)
        _a_ij = e_ij_exp / (e_ij_exp_sum.index_select(0, j_)+ 1e-16)
        #end_time = time.time()
        #print(f"运行时间：{end_time - start_time}秒")
        #print("_a_ij", _a_ij.sum())
        _a_ij = self.dropnet(_a_ij).unsqueeze(-1)
        ## END
        _a_ij_project = e_ij_f * _a_ij
        return _a_ij, _a_ij_project, j_, edge_coord_diff, edge_radial

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
        angle_radial = torch.sqrt(torch.sum(angle_coord_diff**2, 1)).unsqueeze(1)   
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
                              , angle_radial], dim=1)
        angle_ihj_f = self.MLP_atten_angle(angle_ihj_f).view(-1, self.n_heads, self.out_n_node_features)            # 这里需要一个线性层
        # Gat V2
        angle_ihj = self.leakyReLU(angle_ihj_f)
        angle_ihj = (self.a_angle * angle_ihj).sum(-1)
        #angle_ihj = (self.a_angle * angle_ihj_f).sum(-1)
        #angle_ihj = self.leakyReLU(angle_ihj)
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
        return _a_ihj, _a_ihj_project, j_, angle_coord_diff, angle_radial
    
    def forward(self, data: List):
        # 最后一位是squence_attn的平均
        x, edge_index , edge_attr , angle_index, angle_attr, angle_edge_attr, pos, _ = data
        #print("x_size", x.size())
        #start_time = time.time()
        edge_a_, edge_a_project, dst_,  edge_coord_diff, edge_radial = self.atten_edge(x, edge_index, edge_attr, pos)
        x_edge_out = self.atten_update_node(x, edge_a_, edge_a_project, dst_)
        #print("x_edge_out", x_edge_out)
        #print("x", x)
        pos_edge_out = self.atten_update_pos(pos, edge_a_, edge_a_project, dst_, edge_coord_diff, edge_radial)
        #print("IAW[debug NAN1]: pos_edge_out: {}".format(torch.isnan(pos_edge_out).any()))

        #end_time = time.time()
        #print(f"atten_edge运行时间：{end_time - start_time}秒")  
        #start_time = time.time()      
        angle_a_, angle_a_project, dst_, angle_coord_diff, angle_radial = self.atten_angle(x, edge_index, edge_attr, angle_index, angle_attr, angle_edge_attr, pos)
        x_angle_out = self.atten_update_node(x, angle_a_, angle_a_project, dst_)
        #print(angle_coord_diff.size(),edge_coord_diff.size())
        pos_angle_out = self.atten_update_pos(pos, angle_a_, angle_a_project, dst_, angle_coord_diff, angle_radial)
        #end_time = time.time()
        #print(f"atten_angle运行时间：{end_time - start_time}秒")     
        
        # 到这里GAT的消息传递已经完成了
        # 后面按照QKV融合一下

        # sum
        #start_time = time.time()    
        x_edge_out = self.Linear_sum_edge(x_edge_out.view(-1, self.n_heads*self.out_n_node_features)).view(-1, self.n_heads, self.out_n_node_features)                                   #
        #print("sum", x_angle_out.size(), x_angle_out.view(-1, self.n_heads*self.out_n_node_features).size())
        x_angle_out = self.Linear_sum_angle(x_angle_out.view(-1, self.n_heads*self.out_n_node_features)).view(-1, self.n_heads, self.out_n_node_features)                                #
        x_ = self.Linear_sum_x(x).view(-1, self.n_heads, self.out_n_node_features)                                                                                                       #
        
        ## 2-hop注意力计算，分别计算α_ab, α_abc与a之间的注意力分数
        edge_attn = torch.diagonal(torch.matmul(x_, x_edge_out.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.out_n_node_features)
        ## 这里输出一下edge_attn
        #print("IAW[debug]: edge_attn.size = ", edge_attn.size(), edge_attn.sum(), edge_attn.sum(dim=0), edge_attn.sum(dim=-1))
        angle_attn = torch.diagonal(torch.matmul(x_, x_angle_out.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.out_n_node_features)
        squence_attn = torch.stack([edge_attn, angle_attn], dim=0)
        
        # 这是取出edge的注意力
        # 放在这里不行,还是要用softmax的
        #print("squece_attn1", squence_attn.size())
        #squence_attn_mean_list = [squence_attn.mean(dim = 2).unsqueeze(-1)[0,:,:], squence_attn.mean(dim = 2).unsqueeze(-1)[1,:,:]]
        #squence_attn_mean = squence_attn.mean(dim = 2).unsqueeze(-1)[0,:,:]
        #squence_attn_mean = squence_attn.mean(dim = 2).unsqueeze(-1)[1,:,:]

        #iaw = torch.softmax(squence_attn, dim=0).unsqueeze(-1)
        #print(iaw.size(), iaw.sum())
        squence_attn = self.dropnet(torch.softmax(squence_attn, dim=0).unsqueeze(-1))
        #print("squece_attn2", squence_attn.size())
        #print("IAW[debug NAN0]: edge_attn: {}, angle_attn: {}".format(torch.isnan(edge_attn).any(), torch.isnan(angle_attn).any()))

        x_out = squence_attn[0, :, :] * x_edge_out + squence_attn[1, :, :] * x_angle_out
        #print("squence_attn", squence_attn.size(), "x_out", x_out.size())
        
        # 对于pos应该也应atten进行一个融合

        #print("IAW[debug NAN0]: edge_attn: {}, angle_attn: {}".format(torch.isnan(edge_attn).any(), torch.isnan(angle_attn).any()))
        # pos 的sum
        # 这里不能是简单的加和
        #print("pos", pos_edge_out.size())
        #pos_edge_out_ = self.Linear_sum_edge_pos(pos_edge_out)
        #pos_angle_out_ = self.Linear_sum_edge_pos(pos_angle_out)
        #pos_ = self.Linear_sum_edge_pos(pos)

        #edge_attn_pos = torch.diagonal(torch.matmul(pos_, pos_edge_out_.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(3)
        #angle_attn_pos = torch.diagonal(torch.matmul(pos_, pos_angle_out_.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(3)

        #squence_attn_pos = torch.stack([edge_attn_pos, angle_attn_pos], dim=0)
        #squence_attn_pos = torch.softmax(squence_attn_pos, dim=0).unsqueeze(-1)
        #print("squece_attn2", squence_attn.size())
        
        ## 这里输出一下edge_attn
        pos_out = pos_edge_out + pos_angle_out
        #print("pos_out", pos_out)
        #pos_out = squence_attn[0, :, :] * pos_edge_out + squence_attn[1, :, :] * pos_angle_out
        #print("pos", pos_edge_out.size(), pos_out.size(), squence_attn[0,:,:].size(),squence_attn_pos.size())
        #pos_out = pos


        #x_out = x_angle_out 
        #end_time = time.time()
        #print(f"atten_sum运行时间：{end_time - start_time}秒")  
        # 残差Net
        #print("IAW[debug NAN1]: x_out: {}, x: {}".format(torch.isnan(x_out).any(), torch.isnan(x).any()))
        x_out += self.Linear_resnet(x).view(-1, self.n_heads,self.out_n_node_features)                                       #
        #print("IAW[debug NAN2]: x_out: {}".format(torch.isnan(x_out).any()))
        # 合并多头, 采用mean
        #print(x_out.size(), x_out.mean(dim = -1).size(), x_out.mean(dim = 1).size(), self.bias.size())
        x_out = x_out.mean(dim = 1)
        #print("x_out", x_out.size())
        #print("IAW[debug NAN3]: x_out: {}".format(torch.isnan(x_out).any()))
        # bias
        #print("IAW[debug VALUE]: self.bias: {}".format(self.bias))
        x_out += self.bias
        #print("IAW[debug NAN4]: x_out: {}".format(torch.isnan(x_out).any()))
        x_out = self.layer_norm(x_out)                                                                                    # 这里需要一个归一化 
        #print("IAW[debug NAN5]: x_out: {}".format(torch.isnan(x_out).any()))
        x_out = self.activation(x_out)                                                                                    # 这里需要一个归一化层
        #print("IAW[debug NAN6]: x_out: {}".format(torch.isnan(x_out).any()))
        
        #print("squence_attn", x_out.size(), squence_attn.size())
        #squence_attn_mean = squence_attn.mean(dim = 2).mean(dim = 0)
        
        # 这是取出edge的注意力
        #squence_attn_mean = squence_attn.mean(dim = 2)[1,:,:]
        #print("squence_attn_mean", squence_attn_mean)
        # 这是一个很SB的设计
        squence_attn_mean_list = [squence_attn.mean(dim = 2)[0,:,:], squence_attn.mean(dim = 2)[1,:,:]]
        #print("here", squence_attn_mean_list)
        return [x_out, edge_index , edge_attr , angle_index, angle_attr, angle_edge_attr, pos_out, squence_attn_mean_list]


class pred_Layer(nn.Module):
    def __init__(self, dim_per_layer: List, activation=nn.ELU()) -> None:
        super(pred_Layer, self).__init__()
        self.layers = []
        for i in range(len(dim_per_layer) - 2):
            self.layers.append(nn.Linear(dim_per_layer[i], dim_per_layer[i + 1]))
            self.layers.append(activation)
            if i == len(dim_per_layer) - 2 -1:
                self.layers.append(nn.Dropout(0))
        
        self.layers.append(nn.Linear(dim_per_layer[-2], dim_per_layer[-1]))
        
        self.model = nn.Sequential(*self.layers)
        initialize_weights(self.model)

    def forward(self, x: torch.Tensor):
        return self.model(x)

################################################################################
## 需要重新设计expert                                                         ##
## 需要将GNN的内容放在专家中                                                  ##
################################################################################

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 定义输入线性层
        self.input_embedding = nn.Linear(1, hidden_dim)
        
        # 定义查询、键、值的线性层
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, query, key_value):
        """
        query: [batch_size, query_seq_length]
        key_value: [batch_size, key_value_seq_length]
        """
        raw_query = query
        #print(query.shape, key_value.shape)
        # 输入序列可能只有两个维度，所以需要扩展到第三个维度
        query = query.unsqueeze(-1)  # [batch_size, query_seq_length, 1]
        key_value = key_value.unsqueeze(-1)  # [batch_size, key_value_seq_length, 1]
        
        # 进行线性变换扩展到hidden_dim维度
        query = self.input_embedding(query)  # [batch_size, query_seq_length, hidden_dim]
        key_value = self.input_embedding(key_value)  # [batch_size, key_value_seq_length, hidden_dim]
        
        # 获取输入的维度
        batch_size = query.size(0)
        query_seq_length = query.size(1)
        key_value_seq_length = key_value.size(1)
        
        # 线性变换
        query = self.query_linear(query)  # [batch_size, query_seq_length, hidden_dim]
        key = self.key_linear(key_value)  # [batch_size, key_value_seq_length, hidden_dim]
        value = self.value_linear(key_value)  # [batch_size, key_value_seq_length, hidden_dim]
        
        # 计算注意力分数
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, query_seq_length, key_value_seq_length]
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)  # 缩放
        
        # 应用 softmax 归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [batch_size, query_seq_length, key_value_seq_length]
        
        # 加权求和
        output = torch.bmm(attention_probs, value)  # [batch_size, query_seq_length, hidden_dim]
        output = self.output_projection(output).squeeze(-1)
        
        # output = Q + A*V
        output += raw_query
        
        return output, attention_probs




#####################################################################################
## merge_sign: < gnn > 通过交叉注意力机制将分子描述符(md, iaw)对齐到gnn描述符上
##             < iaw > 通过交叉注意力机制将gnn描述符对齐到分子描述符(md, iaw)上
##             < cat > 直接通过cat, 将两部分特征直接拼接在一起
##             < None> 不加入分子描述符
##
#####################################################################################

class MMoE_expert(nn.Module):
    def __init__(self, in_n_node_features_per_layer: List
                     , in_n_angle_features: int
                     , in_n_edge_features: int
                     , n_heads: int
                     , in_n_iaw_features: int
                     , cross_hidden_dim: int
                     , out_dim_expert: int
                     , merge_sign: str):


        super(MMoE_expert, self).__init__()

        layers = [ KAF_Layer(in_n_node_features = in_n_node_features_per_layer[i] 
                           , out_n_node_features= in_n_node_features_per_layer[i+1]
                           , n_heads = n_heads
                           , in_n_angle_features = in_n_angle_features
                           , in_n_edge_features= in_n_edge_features) for i in range(len(in_n_node_features_per_layer) - 1)]

        self.kaf_model = nn.Sequential(*layers)
        

        # gnn_embed
        self.gnn_embed = nn.Sequential(nn.Linear(in_n_node_features_per_layer[-1]*2, in_n_node_features_per_layer[-1]*2)
                                     , nn.LeakyReLU(negative_slope=0.2)
                                     , nn.Linear(in_n_node_features_per_layer[-1]*2, in_n_node_features_per_layer[-1]*2)
                                     , nn.LayerNorm(in_n_node_features_per_layer[-1]*2))
        

       
        self.merge_sign = merge_sign

        # 特征融合之后的维度需要确定
        if merge_sign == "gnn":
            # md_embed
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LeakyReLU(negative_slope=0.2)
                                        , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LayerNorm(in_n_iaw_features))
 
            # 特征融合
            self.merge_embed = CrossAttention(cross_hidden_dim)
            # 特征融合之后的维度
            self.merge_feature_dim = in_n_node_features_per_layer[-1]*2
            # 融合之后的专家输出


        elif merge_sign == "iaw":
            # md_embed
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LeakyReLU(negative_slope=0.2)
                                        , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LayerNorm(in_n_iaw_features))
 
            # 特征融合
            self.merge_embed = CrossAttention(cross_hidden_dim)
            # 特征融合之后的维度
            self.merge_feature_dim = in_n_iaw_features
        
        elif merge_sign == "cat":
            # md_embed
            self.iaw_embed = nn.Sequential(nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LeakyReLU(negative_slope=0.2)
                                        , nn.Linear(in_n_iaw_features, in_n_iaw_features)
                                        , nn.LayerNorm(in_n_iaw_features))
 
            # 特征融合
            self.merge_embed = nn.Sequential(nn.Linear(in_n_node_features_per_layer[-1]*2+in_n_iaw_features
                                                     , in_n_node_features_per_layer[-1]*2+in_n_iaw_features)
                                                     , nn.LeakyReLU(negative_slope=0.2)
                                                     , nn.Linear(in_n_node_features_per_layer[-1]*2+in_n_iaw_features
                                                     , in_n_node_features_per_layer[-1]*2+in_n_iaw_features))

            self.merge_feature_dim = in_n_node_features_per_layer[-1]*2+in_n_iaw_features

        elif merge_sign == "None":
            # 特征融合
            self.merge_embed = nn.Identity()
            self.merge_feature_dim = in_n_node_features_per_layer[-1]*2

        # 融合之后的专家输出
        self.out = nn.Linear(self.merge_feature_dim, out_dim_expert)

    def hidden_embedding(self, data):
        x_out, _, _, _, _, _,_, k_hop_atten_mean = self.kaf_model([data.x, data.edge_index, data.edge_attr, data.angle_index, data.angle_attr, data.angle_edge_attr, data.pos, []])
        
        hop_1_atten, hop_2_atten = k_hop_atten_mean
        # 这里要根据merge_sign 进行不同的处理
        # 不管是哪一种, kaf_model都会计算2hop的注意力
        # 推理的时候batch是None
        if data.batch == None:
            hop_1_atten = [hop_1_atten]
            hop_2_atten = [hop_2_atten]
            #print("k_hop_aten_mean", k_hop_atten_mean)
            pass
        else:
            split_sizes = torch.bincount(data.batch).tolist()
            #k_hop_atten_mean = torch.split(k_hop_atten_mean, split_sizes)
            hop_1_atten = torch.split(hop_1_atten, split_sizes)
            hop_2_atten = torch.split(hop_2_atten, split_sizes)

        #print("hidde_embedding, x_out", x_out.size())
        output1 = global_max_pool(x_out, data.batch)
        output2 = global_mean_pool(x_out , data.batch)

        #print("hidde_embedding after pool, output1", output1.size())
        

        # [(batch_size, n_out), (batch_size, n_out)] -> batch_size, n_out*2
        gnn_embed = self.gnn_embed(torch.cat([output1, output2], dim=1))
        

        # 定义一个变量用于输出, 只是为了写一个return
        attention_matrix = None

        if self.merge_sign != "None":
            # batch_size, n_iaw_attr
            iaw_embed = self.iaw_embed(data.iaw_attr)
            #print("iaw_embed", iaw_embed)
        
        if self.merge_sign == "gnn":
            # 特征融合
            fused_seq, attention_matrix = self.merge_embed(query=gnn_embed, key_value=iaw_embed)

        elif self.merge_sign == "iaw":
            # 特征融合
            fused_seq, attention_matrix = self.merge_embed(query=iaw_embed, key_value=gnn_embed)
        
        elif self.merge_sign == "cat":
            fused_seq = torch.cat([gnn_embed, iaw_embed], dim=1)
            fused_seq = self.merge_embed(fused_seq)

        elif self.merge_sign == "None":
            fused_seq = self.merge_embed(gnn_embed)
        
        # 这里可以考虑加入残差net
        
        # 这里可以考虑再加一个线性层进行变换

        # -> batch_size, n_iaw_attr
        out = self.out(fused_seq)
        
        return out, [hop_1_atten, hop_2_atten, attention_matrix]

    def forward(self, datas):
        data, _ = datas
        embedd1, k_hop_attn = self.hidden_embedding(data)
        #print("embedd1", embedd1.size())        
        return embedd1, k_hop_attn

class MMoE_gate(nn.Module):
    def __init__(self
            , in_dim
            , n_expert
            ):
        super(MMoE_gate, self).__init__()

        self.gate = nn.Linear(in_dim, n_expert)
 
    def forward(self, x):
        return torch.softmax(self.gate(x), dim = 1)

class KAFNet_MD_MMoE_Cross(nn.Module):
    def __init__(self, in_n_node_features_per_layer: List
                      , in_n_angle_features: int
                      , in_n_edge_features: int
                      , n_heads: int
                      , n_pred_hidden_features: int
                      , n_pred_hidden: int
                      , in_n_iaw_features: int
                      , n_experts: int
                      , out_dim_expert: int
                      , cross_hidden_dim: int
                      , merge_sign: str):
        super(KAFNet_MD_MMoE_Cross, self).__init__()

        self.n_experts = n_experts
        
        self.merge_sign = merge_sign
        
        self.mmoe_experts = nn.ModuleList([MMoE_expert(in_n_node_features_per_layer = in_n_node_features_per_layer
                                                     , in_n_angle_features = in_n_angle_features
                                                     , in_n_edge_features = in_n_edge_features
                                                     , n_heads = n_heads
                                                     , in_n_iaw_features = in_n_iaw_features
                                                     , cross_hidden_dim = cross_hidden_dim 
                                                     , out_dim_expert = out_dim_expert
                                                     , merge_sign = merge_sign) for _ in range(n_experts)]) 

        self.mmoe_gates = nn.ModuleList(MMoE_gate(out_dim_expert, n_experts) for _ in range(n_experts))
        
        self.tasks = nn.ModuleList([pred_Layer([out_dim_expert] + [n_pred_hidden_features]*n_pred_hidden +[1]) for _ in range(n_experts)])

    # 分离mmoe,获取中间的embedding
    def mmoe_embedding(self, data):
        
        #expert_k_hop_atten = []
        expert_1_hop_atten = []
        expert_2_hop_atten = []
        
        expert_cross_atten = []
        expert_out = []
        for expert in self.mmoe_experts:
            _out = expert([data, []])
            expert_out.append(_out[0])
            #expert_k_hop_atten.append(_out[1])
            expert_1_hop_atten.append(_out[1][0])
            expert_2_hop_atten.append(_out[1][1])
            expert_cross_atten.append(_out[1][2])

        expert_out = torch.stack(expert_out, dim=1)
        
        #############################################################################################################
        ## 对k_hop_atten, corss_atten进行形状的修改
        ##
        ## expert_k_hop_atten:
        ##         [ [sub_graph1, sub_graph2, sub_grap3, ...]    expert1
        ##          ,[sub_graph1, sub_graph2, sub_grap3, ...]    expert2
        ##          , ...]                                       ...
        _expert_1_hop_atten = []
        _expert_2_hop_atten = []
        _expert_cross_atten = []
        _n_expert = len(expert_1_hop_atten)
        _n_sub_graph = len(expert_1_hop_atten[0])

        #print("expert:", _n_expert, "num sub graph:", _n_sub_graph)
        for j in range(_n_sub_graph):
            _sub_graph_j = []
            for i in range(_n_expert):
                _sub_graph_j.append(expert_1_hop_atten[i][j])

            _sub_graph_j = torch.stack(_sub_graph_j)
            _expert_1_hop_atten.append(_sub_graph_j)

        for j in range(_n_sub_graph):
            _sub_graph_j = []
            for i in range(_n_expert):
                _sub_graph_j.append(expert_2_hop_atten[i][j])

            _sub_graph_j = torch.stack(_sub_graph_j)
            _expert_2_hop_atten.append(_sub_graph_j)

        if self.merge_sign in ["gnn", "iaw"]:
            for j in range(_n_sub_graph):
                _sub_graph_j = []
                for i in range(_n_expert):
                    _sub_graph_j.append(expert_cross_atten[i][j])

                _sub_graph_j = torch.stack(_sub_graph_j)
                _expert_cross_atten.append(_sub_graph_j)
        
        ## 修改之后的形状
        ## _expert_k_hop_atten:
        ##        [   Tensor(n_expert, n_atm, 1), Tensor(n_expert, n_atm, 1), Tensor(n_expert, n_atm, 1), ...]
        ##                   sub_graph1                  sub_graph2                  sub_graph3           ...
        #############################################################################################################
        
        #expert_k_hop_atten = torch.stack(expert_k_hop_atten, dim=1)
        #print("expert_k_hop_atten", expert_k_hop_atten.size())
        
        # batch_size, out_dim_expert
        combined_experts_avg = expert_out.mean(dim=1)  
        
        gate_out = [gate(combined_experts_avg) for gate in self.mmoe_gates]
        # batch_size, n_expert
        gate_out = torch.stack(gate_out, dim=1)
        #for i in range(len(_expert_k_hop_atten)):
        #    print("sub_graph_{}".format(i), _expert_k_hop_atten[i].size())
        #print()
        #print("gate_out", gate_out.size())
        

        #############################################################################################################
        ## 计算sum(g_i * atten_i)
        _batch_size = gate_out.size()[0]
        atten_1_out = []
        atten_2_out = []
        atten_cross_out = []
        #print("gate_out", gate_out)
        for i in range(self.n_experts):
            #_batch_atten = []
            _batch_1_atten = []
            _batch_2_atten = []
            _batch_cross_atten = []
            for b in range(_batch_size):
                # -> (n_expert, 1, 1) * (n_expert, n_atm, 1)

                #_batch_atten.append(torch.sum(gate_out[b, i, :].unsqueeze(-1).unsqueeze(-1) * _expert_k_hop_atten[b], dim = 0))
                #print("Task {}, Graph {}".format(i, b), _batch_atten[-1].size())
                #print("i = ", i, "b = ", b)
                #print(gate_out[b, i, :].unsqueeze(-1).unsqueeze(-1), _expert_1_hop_atten[b])
                _batch_1_atten.append(torch.sum(gate_out[b, i, :].unsqueeze(-1).unsqueeze(-1) * _expert_1_hop_atten[b], dim = 0))
                _batch_2_atten.append(torch.sum(gate_out[b, i, :].unsqueeze(-1).unsqueeze(-1) * _expert_2_hop_atten[b], dim = 0))
                if self.merge_sign in ["gnn", "iaw"]:
                    # 这里单独计算的是cross_atten与gate之间的sum, 因为无法与k_hop_atten相乘
                       
                    _batch_cross_atten.append(torch.sum(gate_out[b, i, :].unsqueeze(-1).unsqueeze(-1) * _expert_cross_atten[b], dim = 0))
                # 其他时候_batch_cross_atten 全是空

            #atten_out.append(_batch_atten)
            atten_1_out.append(_batch_1_atten)
            atten_2_out.append(_batch_2_atten)
            atten_cross_out.append(_batch_cross_atten)
        ## atten_out:
        ## [ [Tensor(n_atm, 1), Tensor(n_atm, 1), ...]         expert1
        ##   [Tensor(n_atm, 1), Tensor(n_atm, 1), ...]         expert2
        ##   [Tensor(n_atm, 1), Tensor(n_atm, 1), ...]         expert3
        ##                                        ...]         expertN
        ##      sub_graph1          sub_graph2    ...
        #print("expert_out", expert_out.size())
        #############################################################################################################


        #############################################################################################################
        ## 计算sum(g_i * v_i)
        ## 计算sum(g_i * y_i)
        task_out = []
        embedd_out = []
        for i in range(self.n_experts):
            #print(expert_out.size(), gate_out.size(), gate_out[:, i, :].size())
            #print("Task {}:".format(i))
            #print("\t\tGate: {}".format(gate_out[:, i, :]))
            weighted_out = torch.sum(gate_out[:, i, :].unsqueeze(-1) * expert_out, dim=1)
            #print(weighted_out.size())
            embedd_out.append(weighted_out)
            #atten_out.append(_atten_out)
            p_out = self.tasks[i](weighted_out)
            #print("iaw3", p_out.size())
            task_out.append(p_out)

        embedd_out = torch.stack(embedd_out, dim=1)
        
        # -> tensor
        task_out = torch.cat(task_out, dim=1)
        #print(atten_out)        
        return task_out, embedd_out, [atten_1_out, atten_2_out, atten_cross_out, gate_out,  _expert_1_hop_atten,  _expert_2_hop_atten,  _expert_cross_atten]

    def forward(self, data):

        task_out, embedd_out, atten_out = self.mmoe_embedding(data)
    
        return task_out 


# 需要有一个单任务模型
class KAFNet_MD_MLP_Cross(nn.Module):
    def __init__(self, in_n_node_features_per_layer: List
                      , in_n_angle_features: int
                      , in_n_edge_features: int
                      , n_heads: int
                      , n_pred_hidden_features: int
                      , n_pred_hidden: int
                      , in_n_iaw_features: int
                      #, n_experts: int
                      , out_dim_expert: int
                      , cross_hidden_dim: int
                      , merge_sign: str):
        super(KAFNet_MD_MLP_Cross, self).__init__()

        
        self.merge_sign = merge_sign
        
        self.expert = MMoE_expert(in_n_node_features_per_layer = in_n_node_features_per_layer
                                                     , in_n_angle_features = in_n_angle_features
                                                     , in_n_edge_features = in_n_edge_features
                                                     , n_heads = n_heads
                                                     , in_n_iaw_features = in_n_iaw_features
                                                     , cross_hidden_dim = cross_hidden_dim 
                                                     , out_dim_expert = out_dim_expert
                                                     , merge_sign = merge_sign)

        
        self.task = pred_Layer([out_dim_expert] + [n_pred_hidden_features]*n_pred_hidden +[1])


    # 分离mmoe,获取中间的embedding
    def expert_embedding(self, data):
        
        _out = self.expert([data, []])

        expert_out = _out[0]
        expert_1_hop_atten = _out[1][0]
        expert_2_hop_atten = _out[1][1]
        expert_cross_atten = _out[1][2]

        task_out = self.task(expert_out)

        
        #print(atten_out)        
        return task_out, expert_out, [expert_1_hop_atten, expert_2_hop_atten, expert_cross_atten]


    def forward(self, data):

        task_out, embedd_out, atten_out = self.expert_embedding(data)
    
        return task_out 


