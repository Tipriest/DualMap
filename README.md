# README.md

## 一. 项目作用
作为`语义地图`/`层次关系图`加载和构建的验证平台
- 输入
  - ROS实时数据流
    - RGB-D消息
    - 机器人位置Pose
    - 相机内参
  - 数据集
    - 数据集
- 输出
  - 地图数据流
    - 语义地图对象列表
    - 语义地图层次关系
- 生成文件
  - layout点云布局
  -
  - 语义地图对象列表
  - 所有类别物体的数量(class_num.json)
- 验证
  - 生成地图之后使用脚本，判断检测率和位置平均检测偏差，最好能够在一张图上打印出来，或者在多张图上打印出来
  -
  - 生成之后查看生成的树状关系图，最好能够用某种方式展示出来，可以用节点拉伸打开这样子
  - 生成之后查看
- 需要负责的任务
  - YOLO扩展与增训
  - 检测准确度检查的脚本和方法
  - 分层的层次地图

- 检测如何提高准确率
  - YOLO的标签可以变得少一点，试一下效果
  - 增训一下YOLO，有增训YOLO的流程
  - 换一下YOLO v8s,用唐立说的大模型DINO来做,会爆显存，可能要调用API modelscope Qwen Yolo的Open Vocabulary系列模型其实也不知有yolo v8 和yoloe，其实还有更多种类的Yolo的模型
  - 后面的提取mask其实可以不使用mobilesam，而是直接使用yolo-seg分割得到的结果
  - rerun上加一个堆栈的图片，显示一下现在的生产-消费者模式的存储队列剩下的内容的数量，可视化的显示一下



## 二. 存在的问题
1. 看一个杂乱一点的地方时间很长很长之后(1000多个keyframe), 内存消耗会爆炸
2. 其实masked出来的东西很多时候有点不对，可能是反向选择了这种
3. 要降低一下现在的CPU使用率，使用率有点大了，要用14-16个核心左右这样子
4. 要看一下内存使用的情况，还是很重要的
5. 这个observations后面也可以是采用固定的长度的一个队列，超过了就随机替换掉一个observations的办法，尝试节省一下内存
6. rerun本身的内存消耗其实也很大

## 三. TODOs
1. 要做一个准确率检测的事情，看一下目前到底检测到了多少东西，召回率和准确率是多少进行判断
2. 想要在rerun里面把局部地图和全局地图拆开，全局地图始终用一个俯视的视角来看，这样能够清楚地看到
全局地图中物体的生成过程，然后全局地图中的物体按照层次关系分成几层，最后通过不同的高度和连线来进行表示



## 构建结果
#### 1. layout.pcd文件
![alt text](assets/layout.png)
#### 2. wall.pcd文件
![alt text](assets/wall.png)



## 安装

> 已在 **Ubuntu 22.04** + **ROS 2 Humble** + **Python 3.10** 上测试通过

#### 1. 克隆仓库（包含子模块）

```bash
git clone --branch main --single-branch --recurse-submodules git@github.com:Tipriest/DualMap.git
cd DualMap
```


#### 2. 创建 Conda 环境
```bash
conda env create -f environment.yml
conda activate dualmap

# 针对特定情况
conda install openssl=3.0.13  # Ubuntu 22.04 常用版本
conda install libcurl
```

#### 3. 安装 MobileCLIP(以后可以安装clip v2)
```bash
cd 3rdparty/mobileclip
pip install -e . --no-deps
cd ../..
```

## 应用

以下是每种应用类型的需求快速概览：

| 应用 | Conda 环境  | ROS2 | Habitat Data Collector |
| :--- | :---: | :---: | :---: |
| 数据集 / 查询 / iPhone | ✓  | | |
| ROS（离线/在线） | ✓ | ✓ | |
| 在线仿真（建图+导航） | ✓ | ✓ | ✓ |


### 💾 使用数据集运行

DualMap 支持使用**离线数据集**运行。当前支持的数据集包括：
1. Replica 数据集
2. ScanNet 数据集
3. TUM RGB-D 数据集
4. 使用 [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) 自行采集的数据

对于从您自己的平台采集的数据，您可以按类似格式组织以运行系统。

遵循[数据集运行指南](resources/doc/app_runner_dataset.md)来安排数据集、使用这些数据集运行 DualMap 并复现我们论文**表 II** 中的离线建图结果。

### 🤖 使用 ROS 运行

DualMap 支持来自 **ROS1** 和 **ROS2** 的输入。您可以使用**离线 rosbags** 或在真实机器人上以**在线模式**运行系统。

遵循 [ROS 运行指南](resources/doc/app_runner_ros.md)开始使用 ROS1/ROS2 rosbags 或实时 ROS 数据流运行 DualMap。

### 🕹️ 仿真中的在线建图与导航

DualMap 通过 [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) 支持仿真中的**在线**交互式建图和物体导航。

遵循[在线建图与导航指南](resources/doc/app_simulation.md)开始在交互式仿真场景中运行 DualMap，并复现我们论文**表 III** 中的导航结果（静态和动态）。

### 📱 使用 iPhone 运行

DualMap 支持从 iPhone 上的 **Record3D** 应用进行**实时数据流传输**。

遵循 [iPhone 运行指南](resources/doc/app_runner_record_3d.md)开始设置 Record3D、将数据流传输到 DualMap，并使用您自己的 iPhone 进行建图！

### 🔍 离线地图查询

我们提供了两个预构建的地图示例用于离线查询：一个来自 iPhone 数据，另一个来自 Replica Room 0。

遵循[离线查询指南](resources/doc/app_offline_query.md)运行查询应用。

### 🖼️ 可视化
<p align="center">
    <img src="resources/image/app_visual.jpg" width="100%">
</p>

系统同时支持 [Rerun](https://rerun.io) 和 [Rviz](http://wiki.ros.org/rviz) 可视化。使用 ROS 运行时，您可以通过 `config/runner_ros.yaml` 中的 `use_rerun` 和 `use_rviz` 选项切换可视化方式。

## 引用

如果您觉得我们的工作有帮助，请考虑为本仓库点星 🌟 并引用：

```bibtex
@article{jiang2025dualmap,
  title={DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes},
  author={Jiang, Jiajun and Zhu, Yiming and Wu, Zirui and Song, Jie},
  journal={arXiv preprint arXiv:2506.01950},
  year={2025}
}
```

## 联系方式
技术问题请创建 issue。其他问题请联系第一作者：jjiang127 [at] connect.hkust-gz.edu.cn

## 致谢

我们感谢 [HOVSG](https://github.com/hovsg/HOV-SG) 和 [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) 作者的贡献和启发。

特别感谢 @[TOM-Huang](https://github.com/Tom-Huang) 在整个项目开发过程中提供的宝贵建议和支持。

我们也感谢 [MobileCLIP](https://github.com/apple/ml-mobileclip)、[CLIP](https://github.com/openai/CLIP)、[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)、[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)、[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) 和 [YOLO-World](https://github.com/AILab-CVC/YOLO-World) 的开发者们提供的优秀开源工作，为本项目提供了强大的技术基础。
