import os
import logging
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
import tensorboard_logger
from tensorboard_logger import log_value
from tqdm import tqdm

from vqapc_model import GumbelAPCModel
from datasets import LibriSpeech


def main():
  parser = argparse.ArgumentParser()

  # RNN architecture config.
  parser.add_argument("--rnn_num_layers", default=3, type=int,
                      help="Number of layers for RNN.")
  parser.add_argument("--rnn_hidden_size", default=512, type=int,
                      help="Hidden size of RNN.")
  parser.add_argument("--rnn_dropout", default=0., type=float,
                      help="RNN dropout rate.")
  parser.add_argument("--rnn_residual", action="store_true",
                      help="Apply residual connections if true.")

  # VQ layer config.
  parser.add_argument("--codebook_size", required=True, type=int,
                      help="Codebook size; all VQ layers will use the same \
                      value.")
  parser.add_argument("--code_dim", default=512, type=int,
                      help="Size of each code.")
  parser.add_argument("--gumbel_temperature", default=0.5, type=float,
                      help="Gumbel-Softmax temperature.")
  parser.add_argument("--vq_hidden_size", default=-1, type=int,
                      help="Hidden size for the VQ layer.")
  parser.add_argument("--apply_VQ", required=True, nargs="+",
                      help="Quantize layer output if 1. E.g., [1, 0, 1] will \
                      apply VQ to the output of the first and third layers.")

  # Optimization config.
  parser.add_argument("--optimizer", default="adam", choices=["adam"],
                      help="Just use adam.")
  parser.add_argument("--batch_size", default=32, type=int,
                      help="Mini-batch size.")
  parser.add_argument("--learning_rate", default=0.0001, type=float,
                      help="Learning rate.")
  parser.add_argument("--epochs", default=100, type=int,
                      help="Number of training epochs.")
  parser.add_argument("--n_future", required=True, type=int,
                      help="Given x_1, ..., x_t, predict x_{t + n_future}.")
  parser.add_argument("--clip_thresh", default=1., type=float,
                      help="Threshold for gradient clipping.")

  # Data config.
  parser.add_argument("--librispeech_home",
                      default="./librispeech_data/preprocessed", type=str,
                      help="Path to the LibriSpeech home directory.")
  parser.add_argument("--train_partition", nargs="+", required=True,
                      help="Partition(s) to be used for training.")
  parser.add_argument("--train_sampling", default=1., type=float,
                      help="Ratio to sample for actual training.")
  parser.add_argument("--val_partition", nargs="+", required=True,
                      help="Partition(s) to be used for validation.")
  parser.add_argument("--val_sampling", default=1., type=float,
                      help="Ratio to sample for actual validation.")

  # Misc config.
  parser.add_argument("--feature_dim", default=80, type=int,
                      help="Dimension of input feature.")
  parser.add_argument("--load_data_workers", default=8, type=int,
                      help="Number of parallel data loaders.")
  parser.add_argument("--exp_name", default="foo", type=str,
                      help="Name of the experiment.")
  parser.add_argument("--store_path", type=str,
                      default="./logs",
                      help="Where to save the trained models and logs.")

  config = parser.parse_args()

  # Create the directory to dump exp logs and models.
  model_dir = os.path.join(config.store_path, config.exp_name + '.dir')
  os.makedirs(config.store_path, exist_ok=True)
  os.makedirs(model_dir, exist_ok=True)

  logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(model_dir, config.exp_name), filemode='w')

  # Define a new Handler to log to console as well.
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

  logging.info('Model Parameters:')
  logging.info('RNN Depth: %d' % (config.rnn_num_layers))
  logging.info('RNN Hidden Dim: %d' % (config.rnn_hidden_size))
  logging.info('RNN Dropout: %f' % (config.rnn_dropout))
  logging.info('RNN Residual Connections: %s' % (config.rnn_residual))
  logging.info('VQ Codebook Size: %d' % (config.codebook_size))
  logging.info('VQ Codebook Dim: %d' % (config.code_dim))
  logging.info('VQ Gumbel Temperature: %f' % (config.gumbel_temperature))
  logging.info('VQ Hidden Dim: %d' % (config.vq_hidden_size))
  apply_VQ = [int(q) > 0 for q in config.apply_VQ]
  logging.info('VQ Apply: %s' % (apply_VQ))
  logging.info('Optimizer: %s' % (config.optimizer))
  logging.info('Batch Size: %d' % (config.batch_size))
  logging.info('Learning Rate: %f' % (config.learning_rate))
  logging.info('Future (n): %d' % (config.n_future))
  logging.info('Gradient Clip Threshold: %f' % (config.clip_thresh))
  logging.info('Training Data: %s' % (config.train_partition))
  logging.info('Training Ratio: %f' % (config.train_sampling))
  logging.info('Validation Data: %s' % (config.val_partition))
  logging.info('Validation Ratio: %f' % (config.val_sampling))
  logging.info('Number of GPUs Used: %d' % (torch.cuda.device_count()))

  model = GumbelAPCModel(input_size=config.feature_dim,
                         hidden_size=config.rnn_hidden_size,
                         num_layers=config.rnn_num_layers,
                         dropout=config.rnn_dropout,
                         residual=config.rnn_residual,
                         codebook_size=config.codebook_size,
                         code_dim=config.code_dim,
                         gumbel_temperature=config.gumbel_temperature,
                         vq_hidden_size=config.vq_hidden_size,
                         apply_VQ=apply_VQ).cuda()
  model = nn.DataParallel(model)

  criterion = nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

  # Setup tensorboard logger.
  tensorboard_logger.configure(
    os.path.join(model_dir, config.exp_name + '.tb_log'))

  # Define data loaders.
  train_set = LibriSpeech(home=config.librispeech_home,
                          partition=config.train_partition,
                          sampling=config.train_sampling)
  # Set drop_last to True to avoid the gather issue when using nn.DataParallel
  train_data_loader = data.DataLoader(train_set, batch_size=config.batch_size,
                                      num_workers=config.load_data_workers,
                                      shuffle=True, drop_last=True)

  val_set = LibriSpeech(home=config.librispeech_home,
                        partition=config.val_partition,
                        sampling=config.val_sampling)
  val_data_loader = data.DataLoader(val_set, batch_size=config.batch_size,
                                    num_workers=config.load_data_workers,
                                    shuffle=False, drop_last=True)

  # Need prefix `module` before state_dict() when using nn.DataParallel.
  torch.save(model.module.state_dict(),
    open(os.path.join(model_dir, config.exp_name + '__epoch_0.model'), 'wb'))

  global_step = 0
  for epoch_i in range(config.epochs):

    ####################
    ##### Training #####
    ####################

    model.train()
    train_losses = []
    for frames_BxLxM, lengths_B in train_data_loader:
      _, indices_B = torch.sort(lengths_B, descending=True)

      frames_BxLxM = Variable(frames_BxLxM[indices_B]).cuda()
      lengths_B = Variable(lengths_B[indices_B]).cuda()

      predicted_BxLxM, _, _ = model(frames_BxLxM[:, :-config.n_future, :],
                                    lengths_B - config.n_future, testing=False)

      optimizer.zero_grad()
      train_loss = criterion(predicted_BxLxM,
                             frames_BxLxM[:, config.n_future:, :])
      train_losses.append(train_loss.item())
      train_loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 config.clip_thresh)
      optimizer.step()

      log_value("training loss (step-wise)", float(train_loss.item()),
                global_step)
      log_value("gradient norm", grad_norm, global_step)

      global_step += 1

    ######################
    ##### Validation #####
    ######################

    model.eval()
    val_losses = []
    with torch.set_grad_enabled(False):
      for val_frames_BxLxM, val_lengths_B in val_data_loader:
        _, val_indices_B = torch.sort(val_lengths_B, descending=True)

        val_frames_BxLxM = Variable(val_frames_BxLxM[val_indices_B]).cuda()
        val_lengths_B = Variable(val_lengths_B[val_indices_B]).cuda()

        val_predicted_BxLxM, _, _ = model(
          val_frames_BxLxM[:, :-config.n_future, :],
          val_lengths_B - config.n_future, testing=True)

        val_loss = criterion(val_predicted_BxLxM,
                             val_frames_BxLxM[:, config.n_future:, :])
        val_losses.append(val_loss.item())

    logging.info('Epoch: %d Training Loss: %.5f Validation Loss: %.5f' % (
      epoch_i + 1, np.mean(train_losses), np.mean(val_losses)))

    log_value("training loss (epoch-wise)", np.mean(train_losses), epoch_i)
    log_value("validation loss (epoch-wise)", np.mean(val_losses), epoch_i)

    torch.save(model.module.state_dict(),
      open(os.path.join(model_dir, config.exp_name + '__epoch_%d' %
      (epoch_i + 1) + '.model'), 'wb'))


if __name__ == '__main__':
  main()
