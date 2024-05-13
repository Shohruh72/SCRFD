### SCRFD is an efficient high accuracy face detection approach
### _Inference code of SCRFD using ONNX Runtime_

![Vizualization](https://github.com/Shohruh72/SCRFD/blob/main/demo/demo.gif)

### Installation

```
conda create -n ONNX python=3.8
conda activate ONNX
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install onnxruntime-gpu==1.14.0
pip install opencv-python==4.5.5.64
```

### Inference
```bash
$ python main.py
```
## Model Performances

|      Name      | Easy  | Medium | Hard  | FLOPs | Params(M) | Infer(ms) |
|:--------------:|-------|--------|-------|-------|-----------|-----------|
|   SCRFD_500M   | 90.57 | 88.12  | 68.51 | 500M  | 0.57      | 3.6       | 
|    SCRFD_1G    | 92.38 | 90.57  | 74.80 | 1G    | 0.64      | 4.1       |
|   SCRFD_2.5G   | 93.78 | 92.16  | 77.87 | 2.5G  | 0.67      | 4.2       |
|   SCRFD_10G    | 95.16 | 93.87  | 83.05 | 10G   | 3.86      | 4.9       |
|   SCRFD_34G    | 96.06 | 94.92  | 85.29 | 34G   | 9.80      | 11.7      |
| SCRFD_500M_KPS | 90.97 | 88.44  | 69.49 | 500M  | 0.57      | 3.6       |
| SCRFD_2.5G_KPS | 93.80 | 92.02  | 77.13 | 2.5G  | 0.82      | 4.3       |
| SCRFD_10G_KPS  | 95.40 | 94.01  | 82.80 | 10G   | 4.23      | 5.0       |

### Note

* This repo supports only inference, see reference for more details

#### Reference

* https://github.com/deepinsight/insightface/tree/master/detection/scrfd
