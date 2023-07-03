# gmot


## Init nvidia-docker
<html>
  <body>
  <summary>Init</summary>
    nvidia-docker run -v /home/lyc/gmot:/home/lyc/gmot -it cddlyf/a100_pytorch1.9.1_cuda11.1:latest /bin/bash
  </body>
</html>

## Installation

<html>
  <summary>Installation</summary>
  
  We use the environment same to DAB-DETR and DN-DETR to run DINO. If you have run DN-DETR or DAB-DETR, you can skip this step. 
  We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well. 

   1. Clone this repo
   ```sh
   https://github.com/yumu-173/gmot.git
   ```

   2. Install other needed packages
    (下载panopticapi需要梯子)
   ```sh
   pip install -r requirements.txt
   ```

   3. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   cd ../../..
   ```
</html>

## Data

<html>
  <summary>Data</summary>

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
Dataset/
  ├── COCO/
    ├── annotations
    ├── train2017
    ├── val2017
  ├── LaSOT/
    ├── airplane
      ├── airplane-1
        ├── img
        └── groundtruth.txt
      ├── ...
    ├── ...
  ├── GOT-10K/
    ├── train_data
      ├── GOT-10K_Train_000001
      ├── ...
    ├── val
  └── instances_coco_lasot_got_train.json
```
</html>

## Run with bash
  <summary>Train with 1 gpu</summary>
  <body>
   bash scripts/DINO_train.sh path/to/COCODIR
  </body>
  <summary>Train with multi gpu</summary>
  <body>
    bash scripts/DINO_train_dist.sh path/to/COCODIR
  </body>
  
  ## Output dir
  <summary>output</summary>
  
  ```
    # output-dir 是main.py中默认路径
    logs/DINO/R50-MS4-1
  ```
