# Background Segmentation PC SDK

A simple background segmentation demo using OpenVINO as the inference engine on a PC.

## Tested On

- **Windows 10 64-bit**
- **OpenVINO 2019_R2**

---

## Code Snippet

```c++
// Initialize the segmentation SDK
int model_index = 0; 
bool force_on_cpu = false; 
std::string cpu_threads_num = "1";
std::string model_folder = "./models/";
SegSdk segSdk(model_index, force_on_cpu, cpu_threads_num, model_folder);

// Prepare your input image (recommended resolution: 480x320, RGB format)
cv::Mat input_img, seg_result;
std::string cvt_format = "RGB"; // or "BGR"

// Perform inference
segSdk.segImg(input_img, seg_result, cvt_format);

