from transformers import BertModel
import torch.nn as nn

class ClfHead(nn.Module):
    ACTIVATIONS = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])
    
    def __init__(self, hid_sizes, num_labels: int, activation: str = 'tanh', dropout: bool = True, dropout_prob: float = 0.3):
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

class SentimentBERT(nn.Module):
    def __init__(self, num_classes):
        super(SentimentBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = ClfHead(self.bert.config.hidden_size, num_classes, dropout_prob=0.5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
