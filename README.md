# Light-NMT
A light-weight attention-based encoder-decoder model for neural machine translation (NMT) in Theano.

This package is based on the [dl4mt](https://github.com/nyu-dl/dl4mt-tutorial).

## Train
```
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 
```
## Test
```
THEANO_FLAGS=device=gpu,floatX=float32 python translate.py 
```
