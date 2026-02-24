"""Transformer模型实现"""

# ================================================
# 导入相关的包
# ================================================
import torch
import torch.nn as nn



# ================================================
# 嵌入层
# ================================================

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

        # 正弦频率插入到位置编码矩阵：对偶数列插入正弦频率
        pe[:, 0::2] = torch.sin(position * div_term)

        # 余弦频率插入到位置编码矩阵：奇数列插入余弦频率
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


# ================================================
# 编码器、解码器
# ================================================

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


class SublayerConnection(nn.Module):
    """
    子层连接：实现残差连接
    流程：上一部分输出 → 层归一化(norm层) → 层归一化输出+上一步分输出 = 残差连接 → 随机失活
    注意：原始transformer结构后进行层归一化。
    """

    def __init__(self, size, dropout):
        """初始化"""

        super(SublayerConnection, self).__init__()

        # 归一化层
        self.norm = LayerNorm(size)

        # 随机失活层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x, sublayer):
        """
        前向传播
        sublayer：为需要进行残差连接的层（例如，多头注意力层，前馈神经网络层）
        """

        # 残差连接
        return x + self.dropout(sublayer(self.norm(x)))

class MutiHeadAttention(nn.Module):
    """
    多头注意力
    Q、K、V → 分别进入三个线性层 → Q、K、V每个输出都转化为多头的形式(向量拆分为多头的形式)
    → 输入Q、K、V计算注意力结果、注意力权重 → 注意力结果经过最后线性层输出

    """
    def __init__(self, h, d_model, dropout=0.1):
        """初始化"""
        super(MutiHeadAttention, self).__init__()

        # 确保能够均分向量多头
        assert d_model % h == 0
        # 头的数量
        self.h = h 
        # 获取每个头维度
        self.d_k = d_model // h
        # 定义多个线性层存入一个列表模块: 四个线性层（前三个线性层分别独立处理Q，V，K，最后一个线性层处理输出）
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        # 注意力
        self.attn = None
        # 失活
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """前向传播"""

        # 如果存在掩码，则将其扩展为三维
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 当前批次的样本数量
        nbatches = query.size(0)

        # 1.执行线性变换:Q、K、V分别进入到各自独立的线性层,并转化为多头的形式(d_model => h x d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for lin, x in zip(self.linears,(query,key,value))
        ]

        # 2.计算注意力和权重(带掩膜)
        x, self.attn = self._attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3.多头注意力合并
        x = (
            x.transpose((1,2)).contiguous().view(nbatches, -1, self.h * self.d_k)
        )

        # 4.最后线性层输出
        x = self.linears[-1](x)

        
        # 返回多头注意力结果
        del query, key, value
        return x

    def _attention(self, query, key, value, mask=None, dropout=None):
        """计算缩放点积注意力"""

        # 每个头的维度
        d_k = query.size(-1)
        # Q、V计算得分
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        # 掩膜处理:遍历循环，将mask中的0位置的元素设置为-1e9
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重/概率:对最后一个维度应用softmax函数归一化为概率分布
        p_attn = scores.softmax(dim=-1)

        # 注意力权重失活
        if drop_out is not None:
            p_attn = dropout(p_attn)

        # 计算注意力输出：注意力权重点乘V
        x = torch.matmul(p_attn, value)

        # 返回注意力
        return x, p_attn


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

class EncoderLayer(nn.Module):
    """
    编码器层: 多头注意力 + 位置化前馈网络
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        # 多头注意力
        self.attn = self_attn

        # 位置化前馈网络
        self.feed_forward = feed_forward 
        
        # 子连接
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

        # 层大小
        self.size = size

    def forward(self, x, mask):
        """
        前向传播
        """

        # 经过多头注意力:最开始的Q、K、V都是相同的,等于x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 经过位置化前馈网络
        x = self.sublayer[1](x, self.feed_forward)

        # 返回结果
        return x
    
class Encoder(nn.module):
    """
    编码器：N个编码器层（多头注意力+位置化前馈层）+ 层归一化
    """
    
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

class Decoderlayer(nn.Module):
    """
    解码器层：两个多头注意力 + 位置化的前馈神经网络
    过程：
        1.解码的注意力计算：输出端的嵌入结果经过第一个注意力层
        2.解码注意力与编码器的结果（memory）进行计算: 解码器的注意力作为Q, memory作为K, V计算融合的注意力结果
        3.将融合的注意力结果作为输入，进行位置化的前馈神经网络
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """初始化"""
        # 继承
        super(Decoderlayer, self).__init__()
        # 层大小
        self.size = size
        # 自注意力层（第一个多头注意力）
        self.self_attn = self_attn
        # 源自注意力层（第二个多头注意力）
        self.src_attn = src_attn
        # 位置化前馈神经网络
        self.feed_forward = feed_forward
        # 子连接层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """前向传播"""

        # 编码器的输出
        m = memory
        # 经过自注意力层(带子层连接)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 经过源自注意力层（带子层连接）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 经过位置化前馈神经网络层
        x = self.sublayer[2](x, self.feed_forward)
        
        return x

class Decoder(nn.Module):
    """
    解码器：n个解码器层+1个正态层
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # n个解码器层
        self.layers = clones(layer, N)
        # 正态层
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        """前向传播"""

        # 经过多层解码器
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        # 经过正态层
        x = self.norm(x)

        return x


# ================================================
# 生成层
# ================================================

class Generator(nn.Module):
    """生成层：从d_model映射到vocab_size"""

    def __init__(self, d_model, vocab):
        """初始化"""
        super(Generator, self).__init__()
        
        # 线性映射
        self.proj = nn.Linear(d_model, vocab)

        # 激活函数
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(slef, x):
        """前向传播"""
        return self.softmax(self.proj(x))



# ================================================
# Transformer
# ================================================

class TransformerModel(nn.Module):
    """Transformer：嵌入层、编码器、解码器、生成层"""

    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        """
        初始化
        src_vocab:源语言的词表大小
        tgt_vocab:目标语言词汇表大小
        N:编码器、解码器的层数
        d_model:嵌入维度
        d_ff:前向传播隐藏层维度
        h:多头注意力的头数
        dropout:丢弃概率
        """

        super(TransformerModel, self).__init__()

        # 嵌入层构建
        position = PositionalEncoding(d_model, dropout)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))

        # 编码器、解码器构建
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

        # 生成层构建：从d_model映射到vocab_size
        self.generation = generation

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
    
    def encode(self, src, src_mask):
        """编码"""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """解码"""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """前向传播：编码、解码、生成"""
        return self.decode(self.encode(src, src_mask),src_mask, tgt, tgt_mask)