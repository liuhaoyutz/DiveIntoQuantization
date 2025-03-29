from typing import Iterable
import torch
import torchvision
from ppq import *
from ppq.api import *

INPUT_SHAPE = [2, 3, 224, 224]
DEVICE = 'cuda'  # only cuda is fully tested :( For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    return [torch.rand(INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model = model.to(DEVICE)

# 在量化配置文件中，ppq 将把指派量配置到模型里。该配置文件以方便后期功能融合解析
setting = QuantizationSettingFactory.pplcuda_setting()
setting.dispatching_table.append(operation="Conv11x1", platform=TargetPlatform.FP32)
setting.dispatching_table.append(operation="Conv25d", platform=TargetPlatform.FP32)


# 量化训练模型(model), calib_dataloader=load_calibration_dataset(), setting=setting,
ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=30, input_shape=INPUT_SHAPE,
    collate_fn=collate_fn)

# 量化误差分析
reports = layerwise_error_analyse(
    graph=ir, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=load_calibration_dataset())

reports = graphwise_error_analyse(
    graph=ir, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=load_calibration_dataset())

# 导出模型，注意导出模型时函数放在最后
# 这是因为我们在 fp.copy() 国际社还不完整的模拟结构 ...
export_ppq_graph(graph=ir, platform=TargetPlatform.ONNXRUNTIME, graph_save_to='quantized_shufflenet.onnx')
