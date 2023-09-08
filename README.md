<hr>
<div align="center">

# Sparsity-Aware Orthogonal Initialization

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/DeepVoltaire/AutoAugment.git"><img alt="Template" src="https://img.shields.io/badge/-AutoAugment-017F2F?style=flat&logo=github&labelColor=gray"></a>


PyTorch implementation of SAO from the paper 
<a href="https://ieeexplore.ieee.org/document/10181312"> Sparsity-Aware Orthogonal Initialization of Deep Neural Networks by Esguerra et al. </a>




</div>

# Overview

Pruning is a common technique to reduce the number of parameters of a deep neural network. However, this technique can have adverse effects such as network disconnection and loss of dynamical isometry.

Thus, our solution is to leverage expander graphs (sparse yet highly-connected graphs) to form the sparse structure and then orthogonalize this structure through appropriate weight assignments.



![alt text](SAO.png)

# Installation

```bash
git clone https://github.com/kiaraesguerra/SAO
cd SAO
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

# Features
* Datasets: CIFAR-10, CIFAR-100, <a href="https://paperswithcode.com/dataset/cinic-10">CINIC-10</a>
* Models: Plain MLP and CNN, <a href="https://arxiv.org/pdf/1512.03385.pdf">ResNet</a>, <a href="https://openreview.net/pdf?id=Zr5W2LSRhD">LipConvNet-N</a>
* Initialization methods: kaiming-normal, <a href="https://arxiv.org/pdf/1806.05393.pdf"> delta-orthogonal initialization</a>, <a href="https://openreview.net/pdf?id=Zr5W2LSRhD"> explicitly-constructed orthogonal convolutions </a>
* Pruning/Sparse construction methods: magnitude pruning, random pruning, Ramanujan pruning, Ramanujan normal, Ramanujan uniform, SAO



# Training

### 1. Delta on Vanilla CNN

```
python main.py --model cnn --num-layers 32 --hidden-width 128 --activation 'relu' --weight-init 'delta' --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

### 2. SAO-Delta on Vanilla CNN

When implementing SAO, the user can specify either the sparsity:

```
python main.py --model cnn --num-layers 32 --hidden-width 128 --activation 'relu' --pruning-method SAO --sparsity 0.5 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

or the degree:

```
python main.py --model cnn --num-layers 32 --hidden-width 128 --activation 'relu' --pruning-method SAO --degree 4 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

Note: When using Tanh, the minimum degree is 2. This should also be noted when specifying the sparsity, such that the sparsity should not result in a degree lower than 2, e.g., for Conv2d(16, 16), the maximum sparsity is 87.50%. For ReLU, the minimum degree is 4, where for Conv2d(16, 16), the maximum sparsity is 75.00%.

### 3. ECO on Vanilla CNN

```
python main.py --model cnn_eco --num-layers 32 --hidden-width 128 --activation 'relu' --weight-init 'delta-eco' --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

### 4. SAO-ECO on Vanilla CNN

Using sparsity:

```
python main.py --model cnn_eco --num-layers 32 --hidden-width 128 --activation 'relu' --pruning-method SAO --sparsity 0.5 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

using degree:

```
python main.py --model cnn_eco --num-layers 32 --hidden-width 128 --activation 'relu' --pruning-method SAO --degree 4 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```


