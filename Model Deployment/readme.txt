Qt平台部署车辆检测与车流量统计模型

如何运行项目代码？
1.下载项目至本地
2.下载、编译依赖库
     ->qt5.13.1 + MSVC2007 64bit
     ->Opencv 4.4.0（Release）
     ->NCNN + Vulkan + protobuf （Release）
     ->Eigen

3.修改外部链接库路径
    ->可以直接按照我的.pro文件路径修改为自己的库文件路径
 
4.修改输入视频、模型权重路径
    ->修改mainwindow.h和ncnnmodelbase.cpp中初始导入视频和模型权重的路径到你的路径。（也可以在运行代码后手动导入）

5.qt creature运行项目代码

   