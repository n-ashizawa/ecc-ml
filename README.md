## PyTorch implementation of \[Repairing Model from Unintended Training with Error Correcting Code\] ##

This demonstrates error-correcting a ResNet18 trained the CIFAR10 dataset.


Install
-----

```
$ git clone git@github.com:fseclab-osaka/ecc-ml.git
$ cd ecc-ml
$ pip install -r requirements.txt
```

Usage
-----

This repository includes the implementation for encoding, decoding, measuring error-correcting rates, and comparing the error-corrected model's inference. 


## Encoding
```
python ecc.py --mode encode
```

## Decoding
```
python ecc.py --mode decode
```

## Mesuring Error-correcting Rates
```
python acc_error_correct.py --mode acc
```

# Comparing Inference Accracy of Error-corrected Model
```
python acc_error_correct.py --mode output
```


# References of Code 

- [Reed-Solomom Codes](https://github.com/tomerfiliba-org/reedsolomon.git)
- Model Pruning: [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://github.com/jacobgil/pytorch-pruning.git)
