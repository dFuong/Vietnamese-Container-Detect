# Vietnamese-Container-Detect

# This repo give pipepline to detecting container following: container truck ->rear container truck

-Should be create virtual enviroment as CUDA, and this following gives manual step in CUDA enviroment

**Create new enviroment**

    $conda create --name video_efficient python=3.7

**Activate your enviroment before installation or running your scripts**

    $conda activate video_efficient

**If you have NVIDIA GPU with cuda 10.1 and cudnn 7.6**

    $pip install tensorflow-gpu==2.2.0

**Installation from anaconda along with cudatoolkit (tf2 version 2.2.0)**

    $conda install -c anaconda tensorflow-gpu==2.2.0
    $conda install -c anaconda tensorflow-gpu

**Clone the tensorflow models responsitory:** [Tensorflow-models-github](https://github.com/tensorflow/models.git)

**Remember to activate your python environment first**

    $cd models/research

**Compile protos**

    $protoc object_detection/protos/*.proto --python_out=.

**Install TensorFlow Object Detection API as a python package**
```
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```


**Install OpenCV python package**
```
pip install opencv-python
pip install opencv-contrib-python
```

**Running video efficientdet**

    $python code/detect_objects.py --video_input --threshold 0.5 --video_path container_3.mp4 --save_output --output_directory out
