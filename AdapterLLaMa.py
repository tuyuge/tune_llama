import torch
from torch import nn
import bmtrain as bmt
import types
import torch.nn.functional as F
from model_center.model.config import LlamaConfig
from model_center.model import Llama


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class BottleneckAdapterLayer(bmt.DistributedModule):
# reference: https://github.com/jain-harshil/Adapter-BERT/blob/master/modeling_adapters.py
    def __init__(self, 
        hidden_size,
        adapter_latent_size,
        dtype=torch.half,
    ):
        super().__init__()
        self.adapter_input_size = hidden_size
        self.adapter_latent_size = adapter_latent_size


        # down projection
        # shape ``(batch, seq_len, dim_ff)``
        self.down_proj = bmt.DistributedParameter(
                torch.empty((self.adapter_latent_size, self.adapter_input_size), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02)
            )
        # up projection
        self.up_proj = bmt.DistributedParameter(
                torch.empty((self.adapter_input_size, self.adapter_latent_size), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02)
            )

        
    def forward(self, x):
        self.act = nn.ReLU()
        output = F.linear(self.act(F.linear(x, self.down_proj)), self.up_proj)
        return output
        

class AdapterLLaMa(nn.Module):
    def __init__(self, plmpath):
        super().__init__()
        self.plmpath = plmpath
        self.plmconfig = LlamaConfig()
        self.model = Llama(self.plmconfig)
        bmt.load(self.model, self.plmpath, strict=False)


        self.adapter_latent_size = 8
        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

        
        self.init_task_delta()
        self.insert_delta()       

        freeze_module(self.model)


    def init_task_delta(self):
        self.adapter_layer = nn.ModuleList([
            BottleneckAdapterLayer(self.plmconfig.dim_model, self.adapter_latent_size)
            for i in range(self.plmconfig.num_layers)
        ])
        bmt.init_parameters(self.adapter_layer)

    
    def insert_delta(self):
        # replace the forward function of FFNBlock object with adapter_linear_forward
        def adapter_linear_forward(
            linear_self,
            hidden_states: torch.Tensor
        ):
            x = linear_self.layernorm_before_ffn(hidden_states)
            if linear_self.post_layer_norm:
                hidden_states = x
            x = linear_self.ffn(x)
            if linear_self.dropout is not None:
                x = linear_self.dropout(x)
            x = self.adapter_layer[linear_self.layer_no](x)
            hidden_states = hidden_states + x
            return hidden_states
        

        for name, module in self.model.named_modules(): 
            if name.endswith(".ffn") and not name.endswith("ffn.ffn"):
                bmt.print_rank("add adapter to", name)
                module.layer_no = int(name.split(".")[-2])
                module.forward = types.MethodType(adapter_linear_forward, module)


    def forward(self, input_ids, length, attention_mask):
        output = self.model(
            input_ids = input_ids,
            length = length,
            attention_mask = attention_mask
        )
        return output