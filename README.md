## PyTorch implementation of \[Repairing Model from Unintended Training with Error Correcting Code\] ##

This demonstrates error-correcting a ResNet18 trained the CIFAR10 dataset.


Install
-----

```
git clone https://github.com/fseclab-osaka/ecc-ml.git
cd ecc-ml
pip install -r requirements.txt
```

Usage
-----

This repository includes the implementation for encoding, decoding, measuring error-correcting rates, and comparing the error-corrected model's inference. 


Encode:
```
python ecc.py --mode encode
```

Decode:
```
python ecc.py --mode decode
```

Mesure Error-correcting Rates:
```
python acc_error_correct.py --mode acc
```

Compare Inference Accracy of Error-corrected Model:
```
python acc_error_correct.py --mode output
```

### Examples
1. Train a ResNet18 model on the CIFAR10 dataset with the over-fitting setting.
```
python train.py --seed 0 --arch resnet18 --dataset cifar10 --lr 0.001 --epoch 100 --over-fitting
```
2. Plot the train and test losses of the trained model
```
python post_losses.py --seed 0 --arch resnet18 --dataset cifar10 --lr 0.001 --epoch 100 --over-fitting
```
3. Select the parameters of the 10th epoch' model for error correction
```
python prune.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20
```
4. Encode 60% of the 32-bit parameters of the 10-th epoch's model with 8-byte redundancy of Reed-Solomon codes 
```
python ecc.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.6 --ecc rs --mode encode --msg-len 32 --t 8
```
5. Decode 60% of the parameters of the 20-th epoch's model to the 10-th epoch's model
```
python ecc.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.6 --ecc rs --mode decode --msg-len 32 --t 8
```
6. Measure the error-correcting rates
```
python acc_error_correct.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.6 --ecc rs --mode acc --msg-len 32 --t 8
```
7. Compare the inference outputs of the models before and after error correction
```
python acc_error_correct.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.1 --ecc rs --mode output --msg-len 32 --t 8
```
8. Count the number of the parameters per their Hamming distances
```
python post_hamming.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.6 --ecc rs --msg-len 32 --t 8
```
9. Post ```.csv``` arranging the results of error correction
```
python post_results.py --seed 0 --arch resnet18 --dataset cifar10 --over-fitting --lr 0.001 --epoch 100 --before 10 --after 20 --target-ratio 0.6 --ecc rs --msg-len 32 --t 8
```


References of Code 
-----

- [Reed-Solomom Codes](https://github.com/tomerfiliba-org/reedsolomon.git)
- Model Pruning: [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://github.com/jacobgil/pytorch-pruning.git)
