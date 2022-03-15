import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalAcc(nn.Module):
  def __init__(self):
    super(CategoricalAcc,self).__init__()
    self.reset()
  def reset(self):
    self._num_correct = 0
    self._num_examples = 0
  
  def forward(self, output):
    y_pred, y = output
    indices = torch.max(y_pred, 1)[1]
    correct = torch.eq(indices, y.permute(1,0)).contiguous().view(-1)
    self._num_correct += torch.sum(correct).item()
    self._num_examples += correct.shape[0]
    return self._num_correct / self._num_examples

cat_acc= CategoricalAcc()

