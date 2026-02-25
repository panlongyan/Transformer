

import time
import torch.distributed as dist # 分布式训练包

class TrainState():
    """训练状态类"""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


class LabelSmoothing(nn.Module):
    """标签平滑类计算损失函数: 用于计算真实标签的概率分布，防止模型过于自信"""
    def __init__(self,size,padding_idx,smoothing=0.0):
        """初始化"""
        super(LabelSmoothing,self).__init__()
        # 创建KL散度损失对象
        self.critertion = nn.KLDivLoss(reduction="sum")
        # 填充索引
        self.padding_idx = padding_idx
        # 平滑参数
        self.smoothing = smoothing
        # 大小
        self.size = size
        # 真实分布
        self.true_dist = None

    def forward(self, x, target):
        """前向传播"""

        # 确保输入的维度是正确的
        assert x.size(1) == self.size

        # 创建一个与x相同现状的张量并初始化为平滑值
        ture_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size-2))

        # 真实标签位置设置为置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 设置填充的位置概率为0
        true_dist[:, self.padding_idx] = 0

        # 处理填充位置的损失
        if mask.dim()>0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        # 计算KL散度值
        return self.critertion(x, self.true_dist.clone.detach())


class SimpleLossCompute:
    """简单损失计算类"""

    def __init__(self, generator, criterion):
        """初始化"""
        # 生成器：transformer中的输出层
        self.generator = generator
        # 损失函数：LabelSmoothing
        self.critertion = LabelSmoothing()

    def __call__(self,x, y, norm):
        """调用损失计算"""

        # 模型输出
        x = self.generator(x)

        # 标准化后损失
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1), y.contiguous().view(-1))
            )
            / norm
        )

        # 返回标准化前、标准化后的损失
        return sloss.data * norm, sloss



class Trainer():
    """训练类"""

    def __init__(self, model):
        """初始化函数"""

        # 模型
        self.model = model
        pass


    def _run_epoch(
        self,
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState()
        ):
        """训练一个epoch"""
        
        # 记录初始时间、总token数、总损失、当前处理token数、梯度累计步数
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0

        # 迭代训练
        for i, batch in enumerate(data_iter):
            # 前向传播
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            # 损失计算:loss_node具有计算图的损失，支持自动微分
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            # 执行反向传播
            if mode == "train" or mode == "train+log":
                # 对损失进行反向传播
                loss_node.backward()
                # 更新信息
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens

                # 梯度累积更新策略：累积多个小批次的梯度后，执行一次参数更新
                if i % accum_iter == 0:
                    # 梯度优化
                    optimizer.step()
                    # 梯度清空
                    optimizer.zero_grad(set_to_none=True)
                    # 已完成梯度累积次数
                    n_accum += 1
                    # 记录总的梯度累积步数
                    train_state.accum_step += 1
                
                # 学习率调度器调整学习率
                scheduler.step()
            
            # 累加总损失、token数量
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            # 打印训练信息
            if i % 40 == 1 and (mode == "train" or mode == "train+log"):
                # 学习率
                lr = optimizer.param_groups[0]["lr"]
                # 时间间隔
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f"
                        + " | Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
    
                # 更新时间
                start = time.time()
                tokens = 0
            
            # 删除loss
            del loss
            del loss_node
        return total_loss / total_tokens, train_state
        
    def _train_worker(
        self,
        gpu,
        ngpus_per_node,
        train_dataloader, 
        valid_dataloader,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
        ):
        """单个训练"""
        def rate(step, model_size, factor, warmup):
            """
            动态调整学习率
            """
            if step == 0:
                step = 1
            return factor * (
                model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
            )


        # 设置GPU设备
        device = torch.ser_device(gpu)

        # 提取填充符号的索引
        pad_idx = vocab_tgt["<blank>"]

        # 隐藏层维度
        d_model = 512

        # 模型放置在GPU设备上
        self.model.cuda(gpu)

        # 模型
        module = self.model

        # 标识当前进程是否为主进程
        is_main_process = True

        # 分布式训练
        if is_distributed:
            # 初始化分布式训练环境
            dist.init_process_group(
                "nccl", # 使用nccl库实现高效的GPU间通信
                init_method="env://", # 使用环境变量初始化分布式训练
                rank=gpu, # 当前进程设置为GPU的编号
                world_size=ngpus_per_node, # 总进程数设置为GPU的数量
            )

            # 将模型绑定到指定GPU
            model = DDP(model, device_ids=[gpu])
            # 模型的模式
            module = model.module
            # 判断当前进程是否为主进程
            is_main_process = gpu == 0
        
        # 定义损失函数
        critertion = LabelSmoothing(
            size=len(vocab_tgt),
            padding_idx=pad_idx,
            smoothing=0.1
        )
        critertion.cuda(gpu)

        # 定义动态优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
        )

        # 定义动态调整优化器学习率对象
        lr_scheduler = LambdaLR(
            optimzer = optimizer,
            lr_lambda = lambda step: rate(
                step, d_model, factor=1, warmup=config["warmup"]
            )
        )

        # 训练状态对象
        train_state = TrainState()

        # for循环迭代训练多个轮次
        for epoch in range(config["num_epochs"]):
            # 启用分布式训练，设置每个轮次的随机种子，确保数据加载随机
            if is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
                valid_dataloader.sampler.set_epoch(epoch)
            
            # 转换为训练模式
            model.train()

            # 执行一个轮次的训练
            _, train_state = _run_epoch(
                (Batch(b[0],b[1],pad_idx) for b in train_dataloader),
                model,
                SimpleLossCompute(modeul.generator, critertion),
                optimizer,
                lr_scheduler,
                mode="train+log",
                accum_iter=config["accum_iter"],
                train_state=train_state
            )

            # 显示GPU使用情况
            GPUtil.showUtilization()
            
            # 判断是否为主进程下进行模型保存（保存模型的状态）
            if is_main_process:
                # 保存地址
                file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
                # 保存模型状态
                torch.save(module.state_dict(), file_path)
            
            # 清空显存
            torch.cuda.empty_cache()

            # 转换评估模式
            model.eval()

            # 获得损失
            sloss = run_epoch(
                (Batch(b[0],b[1],pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(), # 空优化器
                DummyScheduler(), # 空学习率
                mode="eval"
            )
            
            # 清空缓存
            torch.cuda.empty_cache()
        
        # 最后保存模型
        if is_main_process:
            file_path = "%sfinal.pt" % config["file_prefix"]
            torch.save(module.state_dict(), file_path)

    def train_distributed_model(self,):
        """分布式训练：多个训练"""
        pass



    def train(slef):
        """训练函数"""
        pass

