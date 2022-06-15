Python环境下车辆检测模型训练与车流量统计。

本项目使用YOLOX与DeepSORT开源代码完成

YOLOX项目地址：https://github.com/Megvii-BaseDetection/YOLOX
DeepSORT项目地址：https://github.com/ZQPei/deep_sort_pytorch

如何运行项目代码？
1.下载项目至本地

2.安装依赖项

3.训练模型
    ->下载车辆检测数据集并将数据集转换为VOC数据集结构，放到datasets\VOC2007文件夹中。
    ->修改exps\example\yolox_voc中训练集路径、测试集路径、目标类别数。
    ->运行tools\train.py训练你的网络。

4.测试模型
    ->使用tools\demo.py测试你的网络训练效果（按照代码输入parser参数）

5.车流量检测
    ->使用tfd_body.py进行车流量检测 （需要修改你的模型路径、视频路径等）

6.下一步
    ->将pth模型导出为ONNX模型、再使用NCNN工具将ONNC模型导出为NCNN模型（Param、Bin）。
    ->配置qt环境导入模型权重进行推理。
   