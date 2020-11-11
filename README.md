## Vector-Quantized Autoregressive Predictive Coding
For an improved version of VQ-APC, please refer to this [repository](https://github.com/Alexander-H-Liu/NPC).

This repository contains the official implementation of [Vector-Quantized Autoregressive Predictive Coding (VQ-APC)](https://arxiv.org/abs/2005.08392).

VQ-APC is an extension of [APC](https://arxiv.org/abs/1904.03240), which defines a self-supervised task for learning high-level speech representation from unannotated speech. For dependencies and data preprocessing, please refer to the [implementation of APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding). After the data are ready, here's an example command to train your own VQ-APC model:
```
python train_vqapc.py --rnn_num_layers 3 \
                      --rnn_hidden_size 512 \
                      --rnn_dropout 0.1 \
                      --rnn_residual \
                      --codebook_size 128 \
                      --code_dim 512 \
                      --gumbel_temperature 0.5 \
                      --apply_VQ 0 0 1 \
                      --optimizer adam \
                      --batch_size 32  \
                      --learning_rate 0.0001 \
                      --epochs 10 \
                      --n_future 5 \
                      --librispeech_home ./librispeech_data/preprocessed \
                      --train_partition train-clean-360 \
                      --train_sampling 1. \
                      --val_partition dev-clean \
                      --val_sampling 1. \
                      --exp_name my_exp \
                      --store_path ./logs
```
Argument descriptions are available in `train_vqapc.py`.

## TODOs
* Add scripts that get the learned codebook(s) (essentially the parameters of the `nn.Linear` layer used to implement the VQ layers)
* Add scripts that visualize the code-phone co-occurrence (Figure 3 in the paper)

## Reference
Please kindly cite our work if you find this repository useful:
```
@inproceedings{chung2020vqapc,
  title = {Vector-quantized autoregressive predictive coding},
  autohor = {Chung, Yu-An and Tang, Hao and Glass, James},
  booktitle = {Interspeech},
  year = {2020}
}
```

## Contact
You can reach me out via <a href="mailto:andyyuan@mit.edu">email</a>. Questions and feedback are welcome.
