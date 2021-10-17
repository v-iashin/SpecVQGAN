from torch import nn
from .ops.basic_ops import ConsensusModule, Identity
from torch.nn.init import normal, constant
from .bninception.pytorch_load import BNInception

class TSN(nn.Module):
    def __init__(self, modality,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self._prepare_base_model()
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_base_model(self):        
        self.base_model = BNInception()
        self.base_model.last_layer_name = 'fc'
        self.input_size = 224
        self.input_mean = [104, 117, 128]
        self.input_std = [1]

        if self.modality == 'Flow':
            self.input_mean = [128]        
    
    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input):
        sample_len = 3 if self.modality == "RGB" else 2

        input_reshape = input.view((-1, sample_len) + input.size()[-2:])
        base_out = self.base_model(input_reshape)
        if base_out.data.shape[-1] == 1 and base_out.data.shape[-2] == 1:
            base_out = base_out       
        return base_out

    def _construct_flow_model(self, base_model):
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        setattr(container, layer_name, new_conv)
        return base_model