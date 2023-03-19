
This repo contains our solution for the [5th ABAW Challenge](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/). 
We find that [ContraWarping](https://arxiv.org/abs/2303.09034) could extract informative expression features in this in-th-wild large-scale video database.

### Install

This project is based on [MMClassification](https://github.com/open-mmlab/mmclassification), please refer to their repo for installation and dataset preparation.


### Usage

First, download pre-trained weights from [ContraWarping repo](https://github.com/youqingxiaozhua/ContraWarping).

To fine-tune a Res-50 model with two GPUs, run:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    tools/train.py configs/abaw5/ir50.py --launcher pytorch \
    --cfg-options load_from=weights/pretrained_weight.pth # pre-trained weight
```
