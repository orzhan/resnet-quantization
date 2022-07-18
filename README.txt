= ResNet20 quantization experiments =
This repository contains an implementation of ResNet20 training on CIFAR10 dataset, quantization with PyTorch PTQ and my own implementation of quantization. 

= Running =
First, install the dependencies:
`pip install -r requirements.txt`

Then, train the model (or use pre-trained weights):
`python train.py`

The weights will be placed into ./model directory. After that you can run quantization and validate the quantized model:

`python pytorch_ptq.py`

`python ptq_imp.py --bits 8`

Results:
Model | Accuracy | Model Size, MB | Inference speed, samples/s
