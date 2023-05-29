
<h1>Sparsity-Aware Orthogonal Initialization</h2>

PyTorch implementation of SAO from the paper Sparsity-Aware Orthogonal Initialization of Deep Neural Networks by Esguerra et al.

<br>
<h2> 1. Dense Vanilla CNN </h2>







```
python main.py --model van32 --width 128 --activation 'relu' --weight-init 'delta' --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```


<h2> 2. SAO-Delta on Vanilla CNN</h2>

When implement SAO, the user can specify either the sparsity:

```
python main.py --model van32 --width 128 --activation 'relu' --pruning-method SAO --sparsity 0.5 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

or the degree:

```
python main.py --model van32 --width 128 --activation 'relu' --pruning-method SAO --degree 4 --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment True 
```

Note: When using Tanh, the minimum degree is 2. This should also be noted when specifying the sparsity, such that the sparsity should not result in a degree lower than 2, e.g., for Conv2d(16, 16), the maximum sparsity is 87.50%. For ReLU, the minimum degree is 4, where for Conv2d(16, 16), the maximum sparsity is 75.00%.