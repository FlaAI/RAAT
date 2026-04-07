# Robust Alignment: Harmonizing Accuracy and Robustness in Adversarial Training


## 1. Dependencies

```
conda env create -f RAAT.yaml

```


## 2. Training options

- `<DATASET>`: {`cifar10`,`cifar100`,`tinyimagenet`}
- `<ADV_TRAIN OPTION>`: {`adv_train`,`adv_trades`,`adv_mart`}
- `<CONSISTENCY_AT>`: {`consistency`}


## 3. Training code

```
# RAAT Example
python train.py --mode adv_train --BD --epochs 110 --dataset cifar10 --augment_type base

# RAAT++ Example
python train.py --mode adv_mart --BD --epochs 110 --dataset cifar10 --augment_type base

```
