# Streaming People Segmentation PC SDK
**针对直播间场景的视频流下的人像分割，采用openvino作为推理后端。**

所谓“针对直播间场景”，有以下特点：
1. 针对直播间的数据分布，如对坐姿半身的效果最佳（训练数据），舞姿兼容但是效果一般。
2. 后处理算法中利用了摄像头固定的先验，此设置下效果最好。但是也做了移动摄像头的兼容（边缘或会出现抖动等情况）。

根据不同业务对性能与精度的要求，以及GPU的有无，采取栈式的处理策略：
1. 当检测到GPU且force_on_cpu=false时默认使用mode-2
2. 无GPU时依照传入的mode_index进行sdk初始化。

---

**目前支持的模型栈为：**

| Mode Index | Precision          | CPU Usage(%: I7-8700) | Details                                                      |
| ---------- | ------------------ | --------------------- | ------------------------------------------------------------ |
| 0          | :star:             | 1.x                   | 针对弹幕放遮挡等应用，144小分辨率推理和简化的后处理算法      |
| 1          | :star: :star:       | 3.x                   | 针对小框画面等对精度要求一般的应用，如主播游戏画面融合，320分辨率推理和完全的后处理算法。模型与0为同一个模型，但是各自针对特定分辨率finetune过 |
| 2          | :star: :star: :star: | 7.x                   | 针对对精度要求更高的应用，如小程序应用。模型与0/1非一个模型，FLOP更大 |
| GPU mode   | :star: :star: :star: | 1.x                   | OPENVINO的GPU推理0/1/2模型CPU占用均维持在较低水平，因此当检测到GPU时，默认采用2的模型。除非用户强制使用CPU（默认不强制） |

---

**TESTED ON:**

- WINDOWS 10 64x
- OPENVINO  2019_R2

---

**Usage Sample**:

```c++
// seg sdk initialization
int model_index = 0; 
bool force_on_cpu = false; 
std::string cpu_threads_num = "1";
SegSdk segSdk(model_index, force_on_cpu, cpu_threads_num);
// prepare your input img, recomand resolution of (480, 320) and of RGB format
cv::Mat input_img, seg_result;
std::string cvt_format = "RGB"; // "BGR"
// inference
segSdk.segImg(input_img, seg_result, cvt_format)
```



---

# others:
    master分支并非最新分支，最新分支请查看hs_dev分支
