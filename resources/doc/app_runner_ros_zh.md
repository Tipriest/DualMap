# 使用 ROS2 运行


## ROS2 (Humble) 指南

### 环境设置


### 下载 Rosbag 数据

通过 [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) 下载在三个 HM3D 场景中录制的仿真 rosbags。

> **注意：** 如果您已经下载了 `HM3D_collect` 数据集（参见 README 中的[此部分](../../README.md#hm3d-self-collected-data)），则无需再次下载。

[从 OneDrive 下载](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/EqLzgeEJZZVJpttVbDWVDXYBDyiGTMoFB3qaktQONetS6A?e=RJelVv)

| 场景   | 任务     | Rosbag 大小 |
|--------|----------|-------------|
| 00829  | 建图     | 17.2 GB     |
| 00848  | 建图     | 21.2 GB     |
| 00880  | 建图     | 23.3 GB     |


### 使用 ROS2 运行
#### 配置
运行前，请确保配置以下 YAML 文件：

📁 `config/base_config.yaml`

```yaml
# 设置所需的输出目录；建图结果将保存在此处。
output_path: ./output/map_results
```
📁 `config/system_config.yaml`
```yaml
# 根据场景选择合适的类别列表：
# - 对于 HM3D 采集的 rosbag，使用 hm3d300_classes_ycb
# - 对于会议室和走廊：使用 gpt_indoor_office
# - 对于公寓：使用 gpt_indoor_apartment
# - 对于户外：使用 gpt_outdoor_general

given_classes_path: ./config/class_list/hm3d300_classes_ycb.txt
```

📁 `config/support_config/demo_config.yaml`
```yaml
# 世界坐标轴旋转调整
# 修改为默认世界原点
world_roll: 0.0
world_pitch: 0.0
world_yaw: 0.0
```
> 📝 **注意：** 如果您使用我们提供的真实世界 rosbags，此配置对于正确对齐至关重要！
📁 `config/runner_ros.yaml`
```yaml
# 设置 ROS 话题配置
ros_stream_config_path: ./config/data_config/ros/self_collected.yaml

# 是否使用压缩图像话题
use_compressed_topic: false
```

#### 开始运行

**1. 激活 DualMap 环境**

```bash
cd DualMap
conda activate dualmap
```
**2. Source ROS2 并运行 DualMap**


```bash
source /opt/ros/humble/setup.bash
python -m applications.runner_ros
```
您将在终端中看到类似以下的输出：
<p align="center">
    <img src="../image/app_ros/app_ros2_output.jpg" width="70%">
</p>

**4. 启动 ROS2 数据流**

在另一个**新终端**中，开始播放 ROS2 rosbag。这里我们以 **00829** rosbag 为例：

```bash
ros2 bag play path/to/rosbag/rosbag2_odom
```

建图结果将在 **Rerun** 中可视化，如下图所示。  
同时，**全局（抽象）地图**将保存在您配置的 `output_path` 中。


<p align="center">
    <img src="../image/app_ros/app_ros2_00829.jpg" width="70%">
</p>
