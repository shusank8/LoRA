import torch 
import torch.nn as nn
import math
import torch.nn.functional as F


class LoraBase:

    def __init__(self, rank = 8, alpha = 8, dropout = 0):
        self.rank =rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.scailing = self.alpha / ((self.rank) ** (1/2))

    
    def load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict['weight']
        if "bias" in state_dict.keys():
            self.bias.data = state_dict['bias']
        

class LoraLinear(nn.Linear, LoraBase):
    def __init__(self, in_features, out_features, rank, alpha, dropout, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        LoraBase.__init__(self, rank, alpha, dropout)
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))


    def merge_weights(self):
        mw = self.weight + self.scailing * (self.lora_A @ self.lora_B).T
        state_dict = {"weight": mw}
        bias = False
        if self.bias is not None:
            state_dict["bias"] = self.bias
            bias = True

        mdl = nn.Linear(self.in_features, self.out_features, bias)
        mdl.load_state_dict(state_dict)
        return mdl

    def forward(self, x):
        linear_out = x @ self.weight.T
        if self.bias is not None:
            linear_out += self.bias

        lora_mult = self.scailing * (self.lora_A @ self.lora_B)
        lora_out = x @ lora_mult
        return linear_out + lora_out
    

class LoraEmbedding(nn.Embedding, LoraBase):
    def __init__(self, vocab_size, emb_dim, rank, alpha=8, dropout=0):
        nn.Embedding.__init__(self, vocab_size, emb_dim)
        LoraBase.__init__(self, rank, alpha, dropout)
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, emb_dim))

    def merge_weights(self):
        mw = self.weight + self.scailing * (self.lora_A @ self.lora_B)
        state_dict = {"weight": mw}
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict(state_dict)
        return emb_layer

    def forward(self, x):
        emb_out = self.weight[x]

        lora_mult = self.lora_A[x]
        loraout = self.scailing * (lora_mult @ self.lora_B)
        return emb_out + loraout
    

class LoraConv2d(nn.Conv2d, LoraBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        rank=2,
        alpha=2,
        dropout=0,
    ):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, bias
        )
        LoraBase.__init__(self, rank, alpha, dropout)
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_channels, *self.kernel_size)
        )
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def merge_weights(self):
        lora_A_flatten = self.lora_A.flatten(1)
        lora_mult = self.lora_B.T @ lora_A_flatten * self.scailing
        lora_mult = lora_mult.reshape(
            self.out_channels, self.in_channels, *self.kernel_size
        )
        merged_weights = self.weight.data + lora_mult
        state_dict = {"weight": merged_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias
        merged_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True if self.bias is not None else False,
        )
        merged_conv.load_state_dict(state_dict)
        return merged_conv

    def forward(self, x):
        orig_layer_out = F.conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        lora_rank_A_output = F.conv2d(
            x, weight=self.lora_A, stride=self.stride, padding=self.padding
        )
        lora_rank_A_output = lora_rank_A_output.permute(0, 2, 3, 1)

        lora_rank_output = (
            self.dropout(lora_rank_A_output) @ self.lora_B
        ) * self.scailing
        lora_rank_output = lora_rank_output.permute(0, 3, 1, 2)
        return orig_layer_out + lora_rank_output
    

class LoraModel(nn.Module):

    def __init__(self, model, config):
        super(LoraModel, self).__init__()
        self.model = model
        self.config = config
        self.trainable_params = self.compute_parameters_wolora()
        self.disable_all_gradients()
        self.apply_lora(self.model)
        self.toggle_bias()

    def compute_parameters_wolora(self):
        total_param = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_param += param.numel()
        return total_param

    def disable_all_gradients(self):
        for name, param in self.model.named_parameters():
            if any([ex in name for ex in self.config["exclude_modules"]]) is False:
                param.requires_grad = False

    def apply_lora(self, module):
        for name, child in module.named_children():
            if any([tgt in name for tgt in self.config["target_modules"]]) is True:
                if isinstance(child, nn.Linear):
                    new_layer = LoraLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=True if child.bias is not None else False,
                        rank=self.config["rank"],
                        alpha=self.config["alpha"],
                        dropout=self.config["dropout"],
                    )
                    new_layer.load_pretrained_weights(child.state_dict())

                    setattr(module, name, new_layer)

            if (len(list(child.children())) > 0) and not any(
                [ex in name for ex in self.config["exclude_modules"]]
            ):
                self.apply_lora(child)

    def toggle_bias(self):
        for name, param in self.model.named_parameters():
            if not any([ex in name for ex in self.config["exclude_modules"]]):
                if ".bias" in name:
                    if self.config["bias"] == "none":
                        param.requires_grad = False
                    if self.config["bias"] == "all":
                        param.requires_grad = True
                    if (
                        self.config["bias"] == "lora_only"
                        and any([tgt in name for tgt in self.config["target_modules"]])
                        is True
                    ):
                        param.requires_grad = True

    def m_weights(self, module):
        for name, child in module.named_children():
            if isinstance(child, LoraLinear):
                merged_layer = child.merge_weights()
                setattr(module, name, merged_layer)
            if len(list(child.children())) > 0:
                self.m_weights(child)

    def save_model(self, path, merge_wights=False):
        if merge_wights is True:
            self.m_weights(self.model)

            state_dict = {
                name.replace("model.", ""): param.detach().cpu()
                for (name, param) in self.model.named_parameters()
            }

        else:
            state_dict = {
                name: param.detach().cpu()
                for (name, param) in self.model.named_parameters()
                if param.requires_grad
            }

        # TODO: save file implement
        # save_file(state_dict, path)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)