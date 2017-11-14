# Basic Transfer Learning Demo

The dataset Dogs (5000 images) and Cats (5000 images) are provided for easy reproduction of results.

(Inspired by: https://gogul09.github.io/software/flower-recognition-deep-learning)

## Step 1: Clone repo and move to its root dir:

```sh
git clone https://github.com/melvincabatuan/TransferLearningDemo.git
cd TransferLearningDemo
```

## Step 2: Extract Features:

```python
python extract_features.py
```

## Step 3: Train and Test

```python
python train.py
```

## Sample:

```sh

(tensorflow-gpu) D:\>cd TransferLearningDemo

(tensorflow-gpu) D:\TransferLearningDemo>ls

(tensorflow-gpu) D:\TransferLearningDemo>dir
 Volume in drive D is New Volume
 Volume Serial Number is 3E27-7B95

 Directory of D:\TransferLearningDemo

14/11/2017  08:25 AM    <DIR>          .
14/11/2017  08:25 AM    <DIR>          ..
14/11/2017  08:14 AM    <DIR>          config
14/11/2017  08:02 AM    <DIR>          dataset
14/11/2017  08:23 AM             6,004 extract_features.py
14/11/2017  08:02 AM    <DIR>          output
14/11/2017  08:25 AM             4,279 train.py
               2 File(s)         10,283 bytes
               5 Dir(s)  953,004,748,800 bytes free
			   
(tensorflow-gpu) D:\TransferLearningDemo>dir config
 Volume in drive D is New Volume
 Volume Serial Number is 3E27-7B95

 Directory of D:\TransferLearningDemo\config

14/11/2017  08:14 AM    <DIR>          .
14/11/2017  08:14 AM    <DIR>          ..
14/11/2017  08:18 AM               533 inceptionresnetv2_config.json
14/11/2017  08:17 AM               503 inceptionv3_config.json
14/11/2017  08:17 AM               493 mobilenet_config.json
14/11/2017  08:16 AM               488 resnet50_config.json
14/11/2017  08:15 AM               473 vgg16_config.json
14/11/2017  08:15 AM               473 vgg19_config.json
14/11/2017  08:11 AM               488 xception_config.json
               7 File(s)          3,451 bytes
               2 Dir(s)  953,004,744,704 bytes free

(tensorflow-gpu) D:\TransferLearningDemo>dir dataset
 Volume in drive D is New Volume
 Volume Serial Number is 3E27-7B95

 Directory of D:\TransferLearningDemo\dataset

14/11/2017  08:02 AM    <DIR>          .
14/11/2017  08:02 AM    <DIR>          ..
14/11/2017  08:06 AM    <DIR>          dogsvscats
               0 File(s)              0 bytes
               3 Dir(s)  953,004,744,704 bytes free

(tensorflow-gpu) D:\TransferLearningDemo>dir output
 Volume in drive D is New Volume
 Volume Serial Number is 3E27-7B95

 Directory of D:\TransferLearningDemo\output

14/11/2017  08:27 AM    <DIR>          .
14/11/2017  08:27 AM    <DIR>          ..
14/11/2017  08:29 AM    <DIR>          dogsvscats
               0 File(s)              0 bytes
               3 Dir(s)  953,004,744,704 bytes free

(tensorflow-gpu) D:\TransferLearningDemo>dir output\dogsvscats
 Volume in drive D is New Volume
 Volume Serial Number is 3E27-7B95

 Directory of D:\TransferLearningDemo\output\dogsvscats

14/11/2017  08:29 AM    <DIR>          .
14/11/2017  08:29 AM    <DIR>          ..
14/11/2017  08:27 AM    <DIR>          inceptionresnetv2
14/11/2017  08:28 AM    <DIR>          inceptionv3
14/11/2017  08:28 AM    <DIR>          mobilenet
14/11/2017  08:28 AM    <DIR>          resnet50
14/11/2017  08:28 AM    <DIR>          vgg16
14/11/2017  08:29 AM    <DIR>          vgg19
14/11/2017  08:29 AM    <DIR>          xception
               0 File(s)              0 bytes
               9 Dir(s)  953,004,744,704 bytes free
			   
# Xception Example:

# load the user configs
with open('config/xception_config.json') as f:
# with open('config/vgg16_config.json') as f:
# with open('config/vgg19_config.json') as f:
# with open('config/inceptionv3_config.json') as f:
# with open('config/resnet_config.json') as f:
# with open('config/inceptionresnetv2_config.json') as f:
# with open('config/mobilenet_config.json') as f:
	config = json.load(f)


(tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 08:34:26.671795: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 08:34:27.034259: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 08:34:27.036612: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 08:34
[INFO] loading model time: 39.47157526016235
[INFO] successfully loaded base model and model...
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 191.57368421554565
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 4.196108341217041
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 08:38
[INFO] Processing time

 Loading model time: 39.47157526016235
 Feature extraction time: 191.57368421554565
 Feature extraction time per sample: 0.019157368421554567
 
 
 # VGG16
 
 (tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 08:46:44.997817: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 08:46:45.283944: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 08:46:45.286269: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 08:46
[INFO] loading model time: 2.851109027862549
[INFO] successfully loaded base model: vgg16
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 135.86011934280396
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 8.542333602905273
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 08:49
[INFO] Processing time

 Loading model time: 2.851109027862549
 Feature extraction time: 135.86011934280396
 Feature extraction time per sample: 0.013586011934280396

# VGG19

(tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 08:50:56.408352: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 08:50:56.694443: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 08:50:56.697064: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 08:50
[INFO] loading model time: 3.0762686729431152
[INFO] successfully loaded base model: vgg19
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 163.87020349502563
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 8.471522331237793
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 08:53
[INFO] Processing time

 Loading model time: 3.0762686729431152
 Feature extraction time: 163.87020349502563
 Feature extraction time per sample: 0.016387020349502564
 
 # InceptionV3
 
 (tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 08:54:50.285400: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 08:54:50.571701: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 08:54:50.574155: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 08:54
[INFO] loading model time: 8.733557939529419
[INFO] successfully loaded base model: inceptionv3
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 267.92933177948
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 6.760462284088135
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 08:59
[INFO] Processing time

 Loading model time: 8.733557939529419
 Feature extraction time: 267.92933177948
 Feature extraction time per sample: 0.026792933177947998
 
 # ResNet50 
 
 (tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 09:01:30.808415: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 09:01:31.103697: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 09:01:31.106278: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 09:01
[INFO] loading model time: 6.0476014614105225
[INFO] successfully loaded base model: resnet50
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 171.50966906547546
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 16.074846267700195
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 09:04
[INFO] Processing time

 Loading model time: 6.0476014614105225
 Feature extraction time: 171.50966906547546
 Feature extraction time per sample: 0.017150966906547545
 
 # Inception-ResNetV2
 
 (tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 09:05:10.442093: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 09:05:10.729301: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 09:05:10.731770: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 09:05
[INFO] loading model time: 19.2321195602417
[INFO] successfully loaded base model: inceptionresnetv2
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 482.8474853038788
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 5.329460144042969
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 09:13
[INFO] Processing time

 Loading model time: 19.2321195602417
 Feature extraction time: 482.8474853038788
 Feature extraction time per sample: 0.04828474853038788
 
 # MobileNet
 
 (tensorflow-gpu) D:\TransferLearningDemo>python extract_features.py
Using TensorFlow backend.
2017-11-14 09:14:20.243942: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-14 09:14:20.527056: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.61GiB
2017-11-14 09:14:20.529392: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
[STATUS] start time - 2017-11-14 09:14
[INFO] loading model time: 3.1113955974578857
[INFO] successfully loaded base model: mobilenet
[INFO] encoding labels...
[INFO] processed - 0
[INFO] processed - 100
[INFO] processed - 200
[INFO] processed - 300
[INFO] processed - 400
[INFO] processed - 500
[INFO] processed - 600
[INFO] processed - 700
[INFO] processed - 800
[INFO] processed - 900
[INFO] processed - 1000
[INFO] processed - 1100
[INFO] processed - 1200
[INFO] processed - 1300
[INFO] processed - 1400
[INFO] processed - 1500
[INFO] processed - 1600
[INFO] processed - 1700
[INFO] processed - 1800
[INFO] processed - 1900
[INFO] processed - 2000
[INFO] processed - 2100
[INFO] processed - 2200
[INFO] processed - 2300
[INFO] processed - 2400
[INFO] processed - 2500
[INFO] processed - 2600
[INFO] processed - 2700
[INFO] processed - 2800
[INFO] processed - 2900
[INFO] processed - 3000
[INFO] processed - 3100
[INFO] processed - 3200
[INFO] processed - 3300
[INFO] processed - 3400
[INFO] processed - 3500
[INFO] processed - 3600
[INFO] processed - 3700
[INFO] processed - 3800
[INFO] processed - 3900
[INFO] processed - 4000
[INFO] processed - 4100
[INFO] processed - 4200
[INFO] processed - 4300
[INFO] processed - 4400
[INFO] processed - 4500
[INFO] processed - 4600
[INFO] processed - 4700
[INFO] processed - 4800
[INFO] processed - 4900
[INFO] completed label - cats
[INFO] processed - 5000
[INFO] processed - 5100
[INFO] processed - 5200
[INFO] processed - 5300
[INFO] processed - 5400
[INFO] processed - 5500
[INFO] processed - 5600
[INFO] processed - 5700
[INFO] processed - 5800
[INFO] processed - 5900
[INFO] processed - 6000
[INFO] processed - 6100
[INFO] processed - 6200
[INFO] processed - 6300
[INFO] processed - 6400
[INFO] processed - 6500
[INFO] processed - 6600
[INFO] processed - 6700
[INFO] processed - 6800
[INFO] processed - 6900
[INFO] processed - 7000
[INFO] processed - 7100
[INFO] processed - 7200
[INFO] processed - 7300
[INFO] processed - 7400
[INFO] processed - 7500
[INFO] processed - 7600
[INFO] processed - 7700
[INFO] processed - 7800
[INFO] processed - 7900
[INFO] processed - 8000
[INFO] processed - 8100
[INFO] processed - 8200
[INFO] processed - 8300
[INFO] processed - 8400
[INFO] processed - 8500
[INFO] processed - 8600
[INFO] processed - 8700
[INFO] processed - 8800
[INFO] processed - 8900
[INFO] processed - 9000
[INFO] processed - 9100
[INFO] processed - 9200
[INFO] processed - 9300
[INFO] processed - 9400
[INFO] processed - 9500
[INFO] processed - 9600
[INFO] processed - 9700
[INFO] processed - 9800
[INFO] processed - 9900
[INFO] completed label - dogs
[INFO] Feature extraction time: 81.82932829856873
[STATUS] training labels: [0 0 0 ..., 1 1 1]
[STATUS] training labels shape: (10000,)
[INFO] Feature max value: 6.0
[INFO] Feature min value: 0.0
[STATUS] features and labels saved..
[STATUS] end time - 2017-11-14 09:15
[INFO] Processing time

 Loading model time: 3.1113955974578857
 Feature extraction time: 81.82932829856873
 Feature extraction time per sample: 0.008182932829856873
 
 
 # TESTING
 
 (tensorflow-gpu) D:\TransferLearningDemo>python train.py
 [STATUS] start time - 2017-11-14 09:17
 [INFO] Successfully loaded xception features.
[INFO] features shape: (10000, 2048)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 2048)
[INFO] test data   : (2000, 2048)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       0.99      0.99      0.99       990
          1       1.00      1.00      1.00      1010

avg / total       0.99      0.99      0.99      2000

[INFO] prediction time in sec: 0.10935544967651367
[INFO] prediction time per sample: 5.467772483825684e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:23
[INFO] Successfully loaded vgg16 features.
[INFO] features shape: (10000, 4096)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 4096)
[INFO] test data   : (2000, 4096)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       0.95      0.96      0.96       990
          1       0.96      0.95      0.96      1010

avg / total       0.96      0.96      0.96      2000

[INFO] prediction time in sec: 0.140639066696167
[INFO] prediction time per sample: 7.03195333480835e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:24
[INFO] Successfully loaded vgg19 features.
[INFO] features shape: (10000, 4096)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 4096)
[INFO] test data   : (2000, 4096)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       0.95      0.96      0.95       990
          1       0.96      0.95      0.95      1010

avg / total       0.95      0.95      0.95      2000

[INFO] prediction time in sec: 0.14067292213439941
[INFO] prediction time per sample: 7.03364610671997e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:25
[INFO] Successfully loaded inceptionv3 features.
[INFO] features shape: (10000, 2048)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 2048)
[INFO] test data   : (2000, 2048)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       1.00      0.99      1.00       990
          1       1.00      1.00      1.00      1010

avg / total       1.00      1.00      1.00      2000

[INFO] prediction time in sec: 0.10938549041748047
[INFO] prediction time per sample: 5.4692745208740233e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:26
[INFO] Successfully loaded resnet50 features.
[INFO] features shape: (10000, 2048)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 2048)
[INFO] test data   : (2000, 2048)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       0.73      0.77      0.75       990
          1       0.76      0.72      0.74      1010

avg / total       0.74      0.74      0.74      2000

[INFO] prediction time in sec: 0.10938167572021484
[INFO] prediction time per sample: 5.469083786010742e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:27
[INFO] Successfully loaded inceptionresnetv2 features.
[INFO] features shape: (10000, 1536)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 1536)
[INFO] test data   : (2000, 1536)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       1.00      1.00      1.00       990
          1       1.00      1.00      1.00      1010

avg / total       1.00      1.00      1.00      2000

[INFO] prediction time in sec: 0.10938239097595215
[INFO] prediction time per sample: 5.4691195487976076e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>python train.py
[STATUS] start time - 2017-11-14 09:28
[INFO] Successfully loaded mobilenet features.
[INFO] features shape: (10000, 1024)
[INFO] labels shape: (10000,)
[INFO] split into training and testing data...
[INFO] splitted train and test data...
[INFO] train data  : (8000, 1024)
[INFO] test data   : (2000, 1024)
[INFO] train labels: (8000,)
[INFO] test labels : (2000,)
[INFO] loading classifier...
[INFO] creating model/training...
[INFO] dumping classifier...
[INFO] evaluating model...
             precision    recall  f1-score   support

          0       0.99      0.99      0.99       990
          1       0.99      0.99      0.99      1010

avg / total       0.99      0.99      0.99      2000

[INFO] prediction time in sec: 0.10938358306884766
[INFO] prediction time per sample: 5.469179153442383e-05
[INFO] confusion matrix

(tensorflow-gpu) D:\TransferLearningDemo>
```

## Accuracy Results:

- Inception-ResNetv2: 99.65
- InceptionV3: 99.55
- MobileNet : 98.6
- ResNet50 : 74.3
- VGG16 : 95.7
- VGG19 : 95.25
- Xception: 99.5
