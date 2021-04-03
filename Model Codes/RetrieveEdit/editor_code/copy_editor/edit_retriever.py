from torch.nn import Module
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.optim as optim


class EditRetriever(Module):
    def __init__(self, vae_model, ret_model, edit_model):
        super(EditRetriever, self).__init__()
        self.vae_model = vae_model
        self.ret_model = ret_model
        self.edit_model = edit_model
