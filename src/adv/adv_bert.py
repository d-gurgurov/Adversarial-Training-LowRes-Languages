from torch.autograd import Function
import torch.nn as nn
import torch
from typing import Union
from transformers import BertModel


# defining the gradient reversal function and adversarial head
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, lmbda):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lmbda
        return grad_input, None

class ClfHead(nn.Module):
    ACTIVATIONS = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])
    
    def __init__(self, hid_sizes: Union[int, list], num_labels: int, activation: str = 'tanh', dropout: bool = True, dropout_prob: float = 0.3):
        super().__init__()
        
        if isinstance(hid_sizes, int):
            hid_sizes = [hid_sizes]
            out_sizes = [num_labels]
        elif isinstance(hid_sizes, list):
            if len(hid_sizes) == 1:
                out_sizes = [num_labels]
            else:
                out_sizes = hid_sizes[1:] + [num_labels]
        else:
            raise ValueError(f"hid_sizes has to be of type int or list but got {type(hid_sizes)}")
        
        layers = []
        for i, (hid_size, out_size) in enumerate(zip(hid_sizes, out_sizes)):
            if dropout:
                layers.append(nn.Dropout(dropout_prob))
            layers.extend([
                nn.Linear(hid_size, out_size),
                self.ACTIVATIONS[activation]
            ])
        layers = layers[:-1]
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

class AdvHead(nn.Module):
    def __init__(self, adv_count=3, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))
            
    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out
            
    def forward_reverse(self, x, lmbda=1.):
        x_ = ReverseLayerF.apply(x, lmbda)
        return self(x_)
    
    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()

# model architecture
class AdversarialBERT(nn.Module):
    def __init__(self, num_classes_sentiment, num_classes_language, adv_heads=3):
        super(AdversarialBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # sentiment prediction head
        self.sentiment_head = ClfHead(self.bert.config.hidden_size, num_classes_sentiment, dropout=0.5) # TODO: try 0.3 
        
        # language prediction head (adversarial)
        self.language_head = AdvHead(adv_count=adv_heads, hid_sizes=[self.bert.config.hidden_size, num_classes_language], num_labels=num_classes_language, dropout=0.3) # TODO: try 0.3 
        
    def forward(self, input_ids, attention_mask, lmbda):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # sentiment classification
        sentiment_logits = self.sentiment_head(pooled_output)
        
        # language classification (adversarial)
        language_logits = self.language_head.forward_reverse(pooled_output, lmbda=lmbda) # ???? [0]
        
        return sentiment_logits, language_logits
    
    def _get_mean_loss(self, outputs, labels, loss_fn):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        losses = []
        for output in outputs:
            losses.append(loss_fn(output, labels))
        return torch.stack(losses).mean()