def subsequent_mask(size):
    """
    屏蔽后续位置的掩码
    size: 模型每次处理序列的最大长度
    """

    # 注意力矩阵的形状
    attn_shape = (1, size, size)

    # 生成上三角矩阵掩膜（上三角元素为1（代表遮掩），其他元素为0）
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

    # 返回掩膜采用
    return subsequent_mask == 0