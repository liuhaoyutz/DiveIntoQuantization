import torch.profiler
from ppq import *
from ppq.api import *
from tqdm import tqdm

PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

# 生成32个样本输入，每个样本是一个形状为(1, 3, 224, 224)的随机张量
sample_input = [torch.rand(128, 3, 224, 224) for i in range(32)]

# 使用quantize_onnx_model函数对ONNX模型进行量化处理
ir = quantize_onnx_model(
    onnx_import_file='/home/haoyu/work/code/ppq/my_test/Output/resnet18.onnx',  # ONNX模型文件路径
    platform=PLATFORM,
    calib_dataloader=sample_input,  # 校准数据加载器
    calib_steps=16,  # 校准步骤数
    do_quantize=False,  # 是否执行量化
    input_shape=None,  # 输入形状（可选）
    collate_fn=lambda x: x.to('cuda'),  # 数据拼接函数，将数据移动到CUDA设备
    inputs=torch.rand(1, 3, 224, 224).to('cuda')  # 输入张量
)
executor = TorchExecutor(ir)  # 创建TorchExecutor实例

# 使用torch.profiler.profile进行性能分析
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        dir_name='/home/haoyu/work/code/ppq/my_test/working/performance/'
    ),
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    with_stack=True
) as profiler:
    with torch.no_grad():  # 禁用梯度计算以提高性能
        for batch_idx in tqdm(range(10), desc="Profiling ..."):  # 进度条显示
            executor.forward(sample_input[0].to('cuda'))  # 执行前向传播
            profiler.step()  # 更新性能分析器
