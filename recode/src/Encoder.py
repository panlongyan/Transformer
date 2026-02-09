"""编码器"""

import torch.nn as nn
import copy



class Embeddings(nn.Module):
    """嵌入层"""
    def __init__(self, d_model, vocab):
        """初始化"""
        super(Embeddings, self).__init__()
        # 嵌入层：词索引转为稠密向量,vocab:词表大小，d_model:嵌入维度
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        """前向传播"""
        # math.sqrt(self.d_model)作用为向量缩放，使向量长度更接近1
        return self.lut(x) * math.sqrt(self.d_model)

        
class PositionnalEncoding(nn.Module):
    """位置嵌入编码：利用正弦和余弦函数的不同频率来为序列中的每个位置生成唯一的编码"""
    def __init__(self, d_model, dropout, max_len=5000):
        """初始化"""
        super(PositionnalEncoding, self).__init__()

        # 定义失活率
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个矩阵，用于存储位置编码（max_len:处理句子的最大长度）
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引张量:[[0],[1],.......,[max_len-1]]
        position = torch.arange(0, max_len).unsqueeze(1)

        # 决定位置编码中每个维度频率的参数
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 正弦频率插入到位置编码矩阵：对偶数位置插入正弦频率
        pe[:, 0::2] = torch.sin(position * div_term)

        # 余弦频率插入到位置编码矩阵：奇数位置插入余弦频率
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第一个位置扩展为三维
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵存入缓冲区
        self.register_buffer('pe', pe)
    def forward(self, x):
        """前向传播"""

        # self.pe[:, : x.size(1)]仅使用句子长度的位置编码
        # requires_grad_(False):非参数,禁止梯度
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # 返回随机失活后的输出：将部分位置编码的值设置为0
        return self.dropout(x)




class MutiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, h, d_model, dropout=0.1):
        """初始化"""
        super(MutiHeadAttention, self).__init__()








class PositionwiseFeedForward(nn.Module):
    """位置化的前馈神经网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化"""
        super(PositionwiseFeedForward, self).__init__()

        # 线性层1
        self.w_1 = nn.Linear(d_model, d_ff)

        # 线性层2
        self.w_2 = nn.Linear(d_ff, d_model)

        # 激活层
        self.activation = nn.ReLU()

        # 失活层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播"""
        
        # 经过线性层1
        x = self.w_1(x)

        # 经过线性层2
        x = self.w_2(x)

        # 经过激活层
        x = self.activation(x)

        # 经过失活层
        x = self.dropout(x)

        return x





class LayerNorm(nn.Module):
    """
    归一化层:z-score
    引用：https://awesomeml.com/layernorm
    """
    def __init__(self, features, eps=1e-6):
        """初始化"""
        super(LayerNorm, self).__init__()
        # 权重系数
        self.a_2 = nn.Parameter(torch.ones(features))
        # 偏置系数
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 误差项eps
        self.eps = eps
    
    def forward(self, x):
        """前向传播"""

        # 计算均值
        mean = x.mean(-1, keepdim = True)

        # 标准差
        std = x.std(-1, keepdim = True)

        # 归一化: a_2与b_2的作用为增加缩放和平移(增加模型的表达能力), eps为防止零分母
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self,):
        super(EncoderLayer, self).__init__()




class Encoder(nn.module):
    """编码器"""
    def __init__(self,layer, N):
        """初始化"""
        super(Encoder, self).__init__()
        # N层编码层
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        # 层归一化
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        """前向传播"""

        # 经过多层编码层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 层归一化
        return self.norm(x)
