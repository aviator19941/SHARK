import torch_mlir
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import copy
from torch.nn.utils import stateless

import numpy as np
from shark.shark_trainer import SharkTrainer

loss = nn.NLLLoss()

inp = torch.from_numpy(np.load('log_softmax.npy'))
target = torch.from_numpy(np.load('view_12.npy'))
packed_inputs = (inp, target)
output = loss(*packed_inputs)
# output = tensor(5.9949)
print(output)

def train_func(params, buffers, packed_inputs):
    params_and_buffers = {**params, **buffers}
    return stateless.functional_call(
            loss, params_and_buffers, packed_inputs, {}
    )

shark_train = SharkTrainer(loss, packed_inputs)
shark_train.compile(train_func)
params, losses = shark_train.train(1)
# IREE-CPU = array(12.051885, dtype=float32)
