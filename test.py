import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
# 判断是否是主执行文件，如果是，则执行以下代码
if __name__ == "__main__":
    # 启用cuDNN的自动调优器，以选择最适合当前配置的高效算法，提高运算速度
    torch.backends.cudnn.benchmark = True

    # 合并配置文件和命令行参数，获取配置信息和参数
    args, cfg = merge_config()

    # 初始化分布式训练标志
    distributed = False
    # 检查环境变量中是否存在'WORLD_SIZE'，如果存在且大于1，则表示进行分布式训练
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    # 将分布式训练标志保存到配置信息中
    cfg.distributed = distributed

    # 如果是分布式训练
    if distributed:
        # 设置当前进程使用的GPU设备
        torch.cuda.set_device(args.local_rank)
        # 初始化分布式进程组，使用'nccl'后端，并通过环境变量进行初始化
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 根据配置信息获取模型
    net = get_model(cfg)

    # 从指定路径加载预训练模型的参数，并映射到CPU上
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    # 创建一个新的字典，用于存储与当前模型结构兼容的参数
    compatible_state_dict = {}
    # 遍历预训练模型的参数
    for k, v in state_dict.items():
        # 如果参数名中包含'module.'，则去掉前缀，以兼容非分布式训练保存的模型参数
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    # 加载兼容的参数到模型中，并确保模型和参数严格匹配
    net.load_state_dict(compatible_state_dict, strict=True)

    # 如果是分布式训练，则将模型包装为DistributedDataParallel模型，以便在多个GPU上进行并行计算
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # 检查测试工作目录是否存在，如果不存在则创建
    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    eval_lane(net, cfg)