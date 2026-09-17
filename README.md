# TaCo

This repository contains the code for the paper **Learning Flow Semantics for Encrypted Traffic Analysis: A Contrastive Pre-training Approach**, IEEE Transactions on Dependable and Secure Computing (TDSC).

## Pre-training

```
python main_pre.py \
-a TaCo_Encoder -b 8192 \
--optimizer=adamw --lr=1e-4 --weight-decay=.1 \
--epochs=401 --warmup-epochs=20 \
--stop-grad-conv1 --moco-m-cos --moco-t=.2 \
--dist-url 'tcp://localhost:10003' \
--multiprocessing-distributed --world-size 1 --rank 0 \
[data_dir]

python convert_to_deit.py   --input ./checkpoint_0200.pth.tar   --output ./checkpoint_0200.pth
```


## Fine-tuning

Pre-trained encoder: [Link](https://drive.google.com/file/d/16o8mYfk6bq8pePwsZFAScwt2xODtdUDD/view?usp=sharing). Download `pre-trained_TACO.pth` and use its path for `--finetune`.

```
python finetune.py --data_path [data_dir] --nb_classes [number of classses] --finetune [pre-trained_model_dir]
```

## Setup

Dependencies:

- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- timm==0.3.2
- scikit-learn
- matplotlib

## Data

The dataset format is:

```
[Dataset_Path]/[train/test]/[Class]/[Sample]
```

Pre-training dataset:

```
Code: data_process.py
pretrain_MFR_generator([Pcap_Dataset_Path], output_path)
```


fine-tuning dataset:

```
Code: data_process.py
MFR_generator([Pcap_Dataset_Path], output_path)
```
