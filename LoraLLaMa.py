
import torch
from torch import nn
import bmtrain as bmt
import types
import torch.nn.functional as F
import math
from model_center.model.config import LlamaConfig
from model_center.model import Llama


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class LowRankLinear(bmt.DistributedModule):
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        r=8,
        lora_alpha=16,
        dtype=torch.half,
        activation=False
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.activation = activation
        if r > 0:
            self.lora_A = bmt.DistributedParameter(
                torch.empty((r, in_features), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.kaiming_uniform_, a=math.sqrt(5))
            )
            self.lora_B = bmt.DistributedParameter(
                torch.empty((out_features, r), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
            )
            if self.activation:
                self.act = nn.ReLU()
            self.scaling = self.lora_alpha / self.r


    def forward(self, x):
        if self.activation:
            return F.linear(self.act(F.linear(x, self.lora_A)), self.lora_B) * self.scaling
        else:
            return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class LoraLLaMa(nn.Module):
    def __init__(self, plmpath):
        super().__init__()
        self.plmpath = plmpath
        self.plmconfig = LlamaConfig()
        self.model = Llama(self.plmconfig)
        bmt.load(self.model, self.plmpath, strict=False)
        
        self.lora_r = 8

        self.init_task_delta()
        self.insert_delta()

        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        
        freeze_module(self.model)


    def init_task_delta(self):
        self.project_q_lora = nn.ModuleList([
            LowRankLinear(self.plmconfig.dim_model, self.plmconfig.dim_model, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plmconfig.dtype)
            for i in range(self.plmconfig.num_layers)
        ])
        self.project_v_lora = nn.ModuleList([
            LowRankLinear(self.plmconfig.dim_model, self.plmconfig.dim_model, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plmconfig.dtype)
            for i in range(self.plmconfig.num_layers)
        ])
        bmt.init_parameters(self.project_q_lora)
        bmt.init_parameters(self.project_v_lora)
        

    def insert_delta(self):
        def q_lora_linear_forward(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x)

            lora_ret = self.project_q_lora[linear_self.layer_no](x)
            return ret + lora_ret

        def v_lora_linear_forward(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x)
            lora_ret = self.project_v_lora[linear_self.layer_no](x)
            return ret + lora_ret
        

        for name, module in self.model.named_modules(): 
            if name.endswith("self_att.self_attention.project_q"):
                # bmt.print_rank("add lora to", name)
                module.forward_old = module.forward
                module.layer_no = int(name.split(".")[-4])
                module.forward = types.MethodType(q_lora_linear_forward, module)
            elif name.endswith("self_att.self_attention.project_v"):
                # bmt.print_rank("add lora to", name)
                module.forward_old = module.forward
                module.layer_no = int(name.split(".")[-4])
                module.forward = types.MethodType(v_lora_linear_forward, module)


    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            **kwargs
        )
        return output