import os
import torch
from loguru import logger
from deep_sort.deep.models import original_model

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = original_model(751)  # 导入模型
    model.load_state_dict(torch.load(checkpoint)["net_dict"])  # 初始化权重
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")



if __name__ =="__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    checkpoint = './weights/ckpt.t7'
    onnx_path = './featurenet.onnx'
    input = torch.randn(1, 3, 128, 128)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)


'''
ckpt = torch.load(ckpt_file, map_location="cpu")
model.eval()
if "net_dict" in ckpt:
    ckpt = ckpt["net_dict"]
model.load_state_dict(ckpt)

logger.info("loading checkpoint done.")
dummy_input = torch.randn(1, 3, 128,128)

torch.onnx._export(
    model,
    dummy_input,
    output_file,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: 'batch'},
                  "output": {0: 'batch'}},
    opset_version=11,
)
logger.info("generated onnx model named {}".format(output_file))
'''