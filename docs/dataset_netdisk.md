# 数据集处理

从这个[网盘链接](https://t3.znas.cn/Hi7YArimDPf)进行下载，下载后放置在本项目的output路径下:
将压缩文件解压，最终期望的文件路径及结构如下所示:
```bash
cd cur_workspace/output
tree -La3
```
```bash
user at Uranus in ~/dataset
 ○ tree -La 3
.
└── map_results
    ├── hm3d_00829-QaLdnwvtxbs
    │   ├── 20251212_200221
    │   │   ├── detector_time.csv
    │   │   ├── global_map
    │   │   │   ├── global_obj_0000_bathtub
    │   │   │   ├── global_obj_0000_bathtub.pkl
    │   │   │   ├── global_obj_0001_bathtub
    │   │   │   ├── global_obj_0001_bathtub.pkl
    │   │   │   ├── global_obj_0002_tv stand
    │   │   │   ├── global_obj_0002_tv stand.pkl
    │   │   │   ├── global_obj_0003_sink cabinet
    │   │   │   ├── global_obj_0003_sink cabinet.pkl
    │   │   │   ├── global_obj_0004_cabinet
    │   │   │   ├── global_obj_0004_cabinet.pkl
    │   │   │   ├── global_obj_0005_nightstand
    │   │   │   ├── global_obj_0005_nightstand.pkl
    │   │   │   ├── global_obj_0006_floor mat
    │   │   │   ├── global_obj_0006_floor mat.pkl
    │   │   │   ├── global_obj_0007_chair
    │   │   │   ├── global_obj_0007_chair.pkl
    │   │   │   ├── global_obj_0008_bed
    │   │   │   ├── global_obj_0008_bed.pkl
    │   │   │   ├── global_obj_0009_stool
    │   │   │   ├── global_obj_0009_stool.pkl
    │   │   │   ├── global_obj_0010_bench
    │   │   │   ├── global_obj_0010_bench.pkl
    │   │   │   ├── global_obj_0011_desk
    │   │   │   └── global_obj_0011_desk.pkl
    │   │   ├── layout
    │   │   │   └── layout.pcd
    │   │   ├── system_time.csv
    │   │   └── wall
    │   │       └── wall.pcd
    │   ├── detections
    │   └── map

12 directories, 1 file
```