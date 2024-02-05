## PyTorch implementation of \[Repairing Model from Unintended Training with Error Correcting Code\] ##

This demonstrates error-correcting a ResNet18 trained the CIFAR10 dataset.


Usage
-----

This repository includes the implementation for encoding, decoding, measuring error-correcting rates, and comparing the error-corrected model's infetrence. 

Train

......... dogs

......... cats


Test


......... dogs

......... cats


The images were taken from [here](https://www.kaggle.com/c/dogs-vs-cats) but you should try training this on your own data and see if it works!

Training:
`python finetune.py --train`

Pruning:
`python finetune.py --prune`

TBD
---

 - Change the pruning to be done in one pass. Currently each of the 512 filters are pruned sequentually. 
	`
	for layer_index, filter_index in prune_targets:
			model = prune_vgg16_conv_layer(model, layer_index, filter_index)
		`


 	This is inefficient since allocating new layers, especially fully connected layers with lots of parameters, is slow.
	
	In principle this can be done in a single pass.



 - Change prune_vgg16_conv_layer to support additional architectures.
 	The most immediate one would be VGG with batch norm.
