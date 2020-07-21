import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, label_size, d_model, dropout, filter_num, filter_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, d_model)) for fsz in filter_sizes]
        )
        self.linear = nn.Linear(filter_num*len(filter_sizes), label_size)
    def forward(self, input):
        emb = self.embedding(input)
        emb = self.dropout(emb)
        emb = emb.view(emb.size(0), 1, emb.size(1), -1)
        x = [F.relu(conv(emb)) for conv in self.convs]
        x = [F.max_pool2d(x_item, (x_item.size(2), 1)) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        output = self.linear(x)
        return torch.softmax(output, 1)




