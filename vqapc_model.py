import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable


def sample_gumbel(shape, eps=1e-20):
  U = torch.rand(shape).cuda()
  return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
  y = logits + sample_gumbel(logits.size())
  return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
  """From https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
  logits: a tensor of shape (*, n_class)
  returns an one-hot vector of shape (*, n_class)
  """
  y = gumbel_softmax_sample(logits, temperature)
  shape = y.size()
  _, ind = y.max(dim=-1)
  y_hard = torch.zeros_like(y).view(-1, shape[-1])
  y_hard.scatter_(1, ind.view(-1, 1), 1)
  y_hard = y_hard.view(*shape)
  return (y_hard - y).detach() + y


class VQLayer(nn.Module):
  def __init__(self, input_size, hidden_size, codebook_size, code_dim,
               gumbel_temperature):
    """Defines a VQ layer that follows an RNN layer.
      input_size: an int indicating the pre-quantized input feature size,
        usually the hidden size of RNN.
      hidden_size: an int indicating the hidden size of the 1-layer MLP applied
        before gumbel-softmax. If equals to 0 then no MLP is applied.
      codebook_size: an int indicating the number of codes.
      code_dim: an int indicating the size of each code. If not the last layer,
        then must equal to the RNN hidden size.
      gumbel_temperature: a float indicating the temperature for gumbel-softmax.
    """
    super(VQLayer, self).__init__()

    self.with_hiddens = hidden_size > 0

    # RNN hiddens to VQ hiddens.
    if self.with_hiddens:
      # Apply a linear layer, followed by a ReLU and another linear. Following
      # https://arxiv.org/abs/1910.05453
      self.vq_hiddens = nn.Linear(input_size, hidden_size)
      self.relu = nn.ReLU()
      self.vq_logits = nn.Linear(hidden_size, codebook_size)
    else:
      # Directly map to logits without any transformation.
      self.vq_logits = nn.Linear(input_size, codebook_size)

    self.gumbel_temperature = gumbel_temperature
    self.codebook_CxE = nn.Linear(codebook_size, code_dim, bias=False)

  def forward(self, inputs_BxLxI, testing):
    if self.with_hiddens:
      hiddens_BxLxH = self.relu(self.vq_hiddens(inputs_BxLxI))
      logits_BxLxC = self.vq_logits(hiddens_BxLxH)
    else:
      logits_BxLxC = self.vq_logits(inputs_BxLxI)

    if testing:
      # During inference, just take the max index.
      shape = logits_BxLxC.size()
      _, ind = logits_BxLxC.max(dim=-1)
      onehot_BxLxC = torch.zeros_like(logits_BxLxC).view(-1, shape[-1])
      onehot_BxLxC.scatter_(1, ind.view(-1, 1), 1)
      onehot_BxLxC = onehot_BxLxC.view(*shape)
    else:
      onehot_BxLxC = gumbel_softmax(logits_BxLxC,
                                    temperature=self.gumbel_temperature)
    codes_BxLxE = self.codebook_CxE(onehot_BxLxC)

    return logits_BxLxC, codes_BxLxE


class Postnet(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(Postnet, self).__init__()

    input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
    output_sizes = [hidden_size] * num_layers
    self.layers = nn.ModuleList(
      [nn.Linear(in_features=in_size, out_features=out_size)
      for (in_size, out_size) in zip(input_sizes, output_sizes)])

    self.output = (nn.Linear(hidden_size, output_size) if num_layers > 0
                   else nn.Linear(input_size, output_size))

    self.relu = nn.ReLU()

  def forward(self, inputs_BxLxI):
    hiddens_BxLxH = inputs_BxLxI
    for layer in self.layers:
      hiddens_BxLxH = self.relu(layer(hiddens_BxLxH))

    output_BxLxO = self.output(hiddens_BxLxH)

    return output_BxLxO


class GumbelAPCModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, residual,
               codebook_size, code_dim, gumbel_temperature, vq_hidden_size,
               apply_VQ):
    """
      input_size: an int indicating the input feature size, e.g., 80 for Mel.
      hidden_size: an int indicating the RNN hidden size.
      num_layers: an int indicating the number of RNN layers.
      dropout: a float indicating the RNN dropout rate.
      residual: a bool indicating whether to apply residual connections.
      codebook_size: an int indicating the number of codes to learn.
      code_dim: an int indicating the size of each code. Currently must be the
        same as hidden_size.
      gumbel_temperature: a float indicating the temperature for gumbel-softmax.
      vq_hidden_size: an int indicating the hidden size of VQ-layer. If <=0 then
        no intermediate transformation is applied.
      apply_VQ: a list of bools with size `num_layers` indicating whether to
        quantize the output of the corresponding layer. For instance, when
        num_layers equals to 3, a valid apply_VQ would be [True, False, True],
        which will quantize the first and third layer outputs.
    """
    super(GumbelAPCModel, self).__init__()

    assert num_layers > 0
    in_sizes = [input_size] + [hidden_size] * (num_layers - 1)
    out_sizes = [hidden_size] * num_layers
    self.rnn_layers = nn.ModuleList(
      [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
      for (in_size, out_size) in zip(in_sizes, out_sizes)])

    self.rnn_dropout = nn.Dropout(dropout)

    self.rnn_residual = residual

    # VQ layers
    # TODO: Currently code_dim must be the same as RNN hidden_size. Can loose
    # this restriction for the last layer.
    assert hidden_size == code_dim
    self.vq_layers = nn.ModuleList(
      [VQLayer(input_size=hidden_size, hidden_size=vq_hidden_size,
               codebook_size=codebook_size, code_dim=code_dim,
               gumbel_temperature=gumbel_temperature) if q else None
      for q in apply_VQ])
    # TODO: Start with a high temperature and anneal to a small one.

    # Final regression layer
    self.postnet = nn.Linear(code_dim, input_size)

  def forward(self, frames_BxLxM, seq_lengths_B, testing):
    """
    Input:
      frames_BxLxM: a 3d-tensor representing the input features.
      seq_lengths_B: sequence length of frames_BxLxM.
      testing: a bool indicating training or testing phase.

    Return:
      predicted_BxLxM: the predicted output; used for training.
      hiddens_NxBxLxH: the RNN hidden representations across all layers.
      logits_NxBxLxC: logits before gumbel-softmax; used for inferance
        (i.e., pick the largest index as discretized token).
    """
    max_seq_len = frames_BxLxM.size(1)

    # N is the number of RNN layers.
    hiddens_NxBxLxH = []
    logits_NxBxLxC = []

    # RNN
    # Prepare initial packed RNN input.
    packed_rnn_inputs = pack_padded_sequence(frames_BxLxM, seq_lengths_B,
                                             batch_first=True,
                                             enforce_sorted=False)
    for i, (rnn_layer, vq_layer) in enumerate(
      zip(self.rnn_layers, self.vq_layers)):
      # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/14
      rnn_layer.flatten_parameters()
      packed_rnn_outputs, _ = rnn_layer(packed_rnn_inputs)

      # Unpack RNN output of current layer.
      rnn_outputs_BxLxH, _ = pad_packed_sequence(packed_rnn_outputs,
                                                 batch_first=True,
                                                 total_length=max_seq_len)
      # Apply dropout to output.
      rnn_outputs_BxLxH = self.rnn_dropout(rnn_outputs_BxLxH)

      # Apply residual connections.
      if self.rnn_residual and i > 0:
        # Unpack the original input.
        rnn_inputs_BxLxH, _ = pad_packed_sequence(packed_rnn_inputs,
                                                  batch_first=True,
                                                  total_length=max_seq_len)
        rnn_outputs_BxLxH += rnn_inputs_BxLxH

      hiddens_NxBxLxH.append(rnn_outputs_BxLxH)

      if vq_layer is not None:
        logits_BxLxC, rnn_outputs_BxLxH = vq_layer(rnn_outputs_BxLxH, testing)
        logits_NxBxLxC.append(logits_BxLxC)
      else:
        logits_NxBxLxC.append(None)

      # Prepare packed input for the next layer.
      packed_rnn_inputs = pack_padded_sequence(rnn_outputs_BxLxH,
                                               seq_lengths_B, batch_first=True,
                                               enforce_sorted=False)
    hiddens_NxBxLxH = torch.stack(hiddens_NxBxLxH)

    # Generate final output from codes.
    predicted_BxLxM = self.postnet(rnn_outputs_BxLxH)

    return predicted_BxLxM, hiddens_NxBxLxH, logits_NxBxLxC
