import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dni

def one_hot(indexes, n_classes, device='cpu'):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    result.to(device)
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank,
        index=indexes.data.unsqueeze(dim=indexes_rank),
        value=1
    )
    return Variable(result)


class Net(nn.Module):
    def __init__(self, use_dni=False, context=False, device='cpu', freeze_synthesizer=False, trained_net_file=None):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(784, 32, bias=False)
        self.hidden1_bn = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 32, bias=False)
        self.hidden2_bn = nn.BatchNorm1d(32)

        self.use_dni = use_dni
        self.context = context
        self.device = device
        self.freeze_synthesizer = freeze_synthesizer

        if self.use_dni:
            if self.context:
                context_dim = 10
            else:
                context_dim = None
            self.backward_interface = dni.BackwardInterface(
                dni.BasicSynthesizer(
                    output_dim=32, context_dim=context_dim
                )
            )
            if trained_net_file is not None:
                trained_net = torch.load(trained_net_file)
                dict_trained_params = dict(trained_net.backward_interface.named_parameters())
                for name, param in self.backward_interface.named_parameters():
                    if name in dict_trained_params:
                        param.requires_grad = False
                        print('Copying pretrained ' + name)
                        param.copy_(dict_trained_params[name].data)
                        param.requires_grad = True
                
            if self.freeze_synthesizer:
                for name, param in self.backward_interface.named_parameters():
                    param.requires_grad = False
        self.output = nn.Linear(32, 10, bias=False)
        self.output_bn = nn.BatchNorm1d(10)
        self.to(device)

    def forward(self, x, y=None):
        x = x.view(x.size()[0], -1)
        x = self.hidden1_bn(self.hidden1(x))
        x = self.hidden2_bn(self.hidden2(F.relu(x)))
        if self.use_dni and self.training:
            if self.context:
                context = one_hot(y, 10, self.device)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
        x = self.output_bn(self.output(F.relu(x)))
        return F.log_softmax(x)