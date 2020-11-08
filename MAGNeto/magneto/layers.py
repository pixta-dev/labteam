import math
import copy

import torch
import torch.nn as nn
from torchvision import models


def freeze_all_parameters(module: nn.Module):
    ''' Freeze all parameters of a PyTorch Module.
    input:
        + module: self-explanatory.
    '''
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_all_parameters(module: nn.Module):
    ''' Unfreeze all parameters of a PyTorch Module.
    input:
        + module: self-explanatory.
    '''
    for param in module.parameters():
        param.requires_grad = True


class TagEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TagEmbedder, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size+1,  # Plus the padding
            embedding_dim=d_model,
        )

    def forward(self, x):
        return self.embed(x)


class MultiHeadMaskedScaledDotProduct(nn.Module):
    def __init__(self, d_k: int):
        '''
        input:
            + d_k: the dimensionality of the subspace.
        '''
        super(MultiHeadMaskedScaledDotProduct, self).__init__()

        self.d_k = d_k

    def forward(
        self,
        q: torch.tensor,
        k: torch.tensor,
        mask: torch.tensor = None
    ) -> torch.tensor:
        '''
        input:
            + q: the matrix of queries.
            + k: the matrix of keys.
            + mask: used to mask out padding positions.
        output:
            The matrix of scores.
        '''
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        '''
        input:
            + heads: the number of heads of each Multi-Head Attention layer.
            + d_model: the dimentionality of a context vector, must be divisible by the number of heads.
            + dropout: dropout value of tag encoder layers.
        '''
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dp = MultiHeadMaskedScaledDotProduct(self.d_k)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.tensor,
        k: torch.tensor,
        v: torch.tensor,
        mask: torch.tensor = None
    ) -> torch.tensor:
        '''
        input:
            + q: the matrix of queries.
            + k: the matrix of keys.
            + v: the matrix of values.
            + mask: used to mask out padding positions.
        output:
            context vectors.
        '''
        bs = q.size(0)

        # Perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # Transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.dp(q, k, mask)
        scores = self.softmax(scores)

        scores = self.dropout(scores)

        # Compute context vectors based on calculated scores above
        context = torch.matmul(scores, v)

        # Concatenate heads and put through final linear layer
        concat = context.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class TagToImageLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        '''
        input:
            + d_model: the dimentionality of a context vector, must be divisible by the number of heads.
            + heads: the number of heads of the Multi-Head Attention sub-layer.
            + dropout: the dropout value of the Multi-Head Attention sub-layer.
        '''
        super(TagToImageLayer, self).__init__()

        self.attn = MultiHeadAttention(
            heads, d_model, dropout=dropout)

    def forward(self, tag_vectors: torch.tensor, img_regions: torch.tensor) -> torch.tensor:
        '''
        input:
            + tag_vectors: self-explanatory.
            + img_regions: self-explanatory.
        output:
            output vectors.
        '''
        out = self.attn(tag_vectors, img_regions, img_regions)

        return out


class GatingLayer(nn.Module):
    def __init__(self, in_features: int, dim_feedforward: int, dropout: float = 0.1):
        '''
        input:
            + in_features: the dimentionality of the input vectors.
            + dim_feedforward: the dimentionality of the hidden layer.
            + dropout: the dropout value of the Gating layer.
        '''
        super(GatingLayer, self).__init__()

        self.dropout_1 = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_features, dim_feedforward)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tag_vectors: torch.tensor) -> torch.tensor:
        '''
        input:
            + tag_vectors: self-explanatory.
        output:
            output gating values.
        '''
        out = self.dropout_1(tag_vectors)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.dropout_2(out)
        out = self.linear_2(out)
        out = self.sigmoid(out.squeeze(dim=-1))

        return out


class ImageFeatureExtractor(nn.Module):
    def __init__(self, d_model: int, img_backbone: str):
        '''
        input:
            + d_model: the dimentionality of a context vector, must be divisible by the number of heads.
        '''
        super(ImageFeatureExtractor, self).__init__()

        if img_backbone == 'resnet18':
            encoder = models.resnet18(pretrained=True)
            ex_out_dim = 512
        elif img_backbone == 'resnet50':
            encoder = models.resnet50(pretrained=True)
            ex_out_dim = 2048

        # Get all layers
        encoder_children = list(encoder.children())
        # Drop the last avg & fc layers
        self.backbone = nn.Sequential(*encoder_children[:-2])

        self.conv_1x1 = nn.Conv2d(ex_out_dim, d_model, kernel_size=(
            1, 1), stride=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(d_model)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.freeze_all_layers()
        self.unfreeze_top_layers()
        # self.unfreeze_the_fourth_block()
        # self.unfreeze_the_third_block()
        # self.unfreeze_the_second_block()
        # self.unfreeze_the_first_block()
        # self.unfreeze_the_bottom_layers()

    def freeze_all_layers(self):
        ''' Freeze all image encoder's layers.
        '''
        freeze_all_parameters(self)

    def unfreeze_top_layers(self):
        ''' Unfreeze the top layers of the image feature extractor.
        '''
        # conv_1x1
        unfreeze_all_parameters(self.conv_1x1)

        # bn
        unfreeze_all_parameters(self.bn)

    def unfreeze_the_first_block(self):
        ''' Unfreeze the first block of the image encoder.
        '''
        assert type(self.backbone[4]) is nn.Sequential

        unfreeze_all_parameters(self.backbone[4])

    def unfreeze_the_second_block(self):
        ''' Unfreeze the second block of the image encoder.
        '''
        assert type(self.backbone[5]) is nn.Sequential

        unfreeze_all_parameters(self.backbone[5])

    def unfreeze_the_third_block(self):
        ''' Unfreeze the third block of the image encoder.
        '''
        assert type(self.backbone[6]) is nn.Sequential

        unfreeze_all_parameters(self.backbone[6])

    def unfreeze_the_fourth_block(self):
        ''' Unfreeze the fourth block of the image encoder.
        '''
        assert type(self.backbone[7]) is nn.Sequential

        unfreeze_all_parameters(self.backbone[7])

    def unfreeze_the_bottom_layers(self):
        ''' Unfreeze the bottom layers of the image encoder.
        '''
        # Unfreeze the first conv layer and bn
        assert type(self.backbone[0]) is nn.Conv2d
        assert type(self.backbone[1]) is nn.BatchNorm2d
        assert type(self.backbone[2]) is nn.ReLU
        assert type(self.backbone[3]) is nn.MaxPool2d

        # conv
        unfreeze_all_parameters(self.backbone[0])

        # bn
        unfreeze_all_parameters(self.backbone[1])

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        input:
            + x: input image.
        output:
            image's features.
        '''
        features = self.backbone(x)

        out = self.conv_1x1(features)
        out = self.bn(out)

        # Convert the tensor from channel first to channel last
        out = out.permute(0, 2, 3, 1)

        out = self.flatten(out)

        return out
