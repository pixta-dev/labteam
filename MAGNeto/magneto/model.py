import torch
import torch.nn as nn

from magneto.layers import (
    TagEmbedder,
    ImageFeatureExtractor,
    TagToImageLayer,
    GatingLayer
)


class MAGNeto(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        t_blocks: int,
        t_heads: int,
        i_blocks: int,
        i_heads: int,
        dropout: float,
        t_dim_feedforward: int = 2048,
        i_dim_feedforward: int = 2048,
        g_dim_feedforward: int = 2048,
        img_backbone: str = 'resnet50',
    ):
        '''
        input:
            + d_model: the dimentionality of a context vector, must be divisible by the number of heads.
            + vocab_size: self explanatory.
            + t_blocks: the number of encoder layers, or blocks, for tag branch.
            + t_heads: the number of heads of each Multi-Head Attention layer of the tag branch.
            + i_blocks: the number of encoder layers, or blocks, for image branch.
            + i_heads: the number of heads of each Multi-Head Attention layer of the image branch.
            + dropout: dropout value of the whole network.
            + t_dim_feedforward: the dimension of the feedforward network model in the TransformerEncoderLayer class of the tag branch.
            + i_dim_feedforward: the dimension of the feedforward network model in the TransformerEncoderLayer class of the image branch.
            + g_dim_feedforward: the dimension of the feedforward network model in the GatingLayer class.
            + img_backbone: resnet18 or resnet50.
        '''
        super(MAGNeto, self).__init__()

        self.tag_embedder = TagEmbedder(vocab_size, d_model)
        self.tag_dropout = nn.Dropout(dropout)
        self.tag_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=t_heads, dim_feedforward=t_dim_feedforward, dropout=dropout),
            num_layers=t_blocks
        )
        self.tag_linear = nn.Linear(d_model, 1)
        self.tag_sigmoid = nn.Sigmoid()

        self.img_feature_extractor = ImageFeatureExtractor(
            d_model, img_backbone)
        self.tag_to_img = TagToImageLayer(d_model, i_heads, dropout)
        self.img_dropout = nn.Dropout(dropout)
        self.img_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=i_heads, dim_feedforward=i_dim_feedforward, dropout=dropout),
            num_layers=i_blocks
        )
        self.img_linear = nn.Linear(d_model, 1)
        self.img_sigmoid = nn.Sigmoid()

        self.gating = GatingLayer(
            d_model * 2, dim_feedforward=g_dim_feedforward, dropout=dropout)

    def forward(self, src: torch.tensor, img: torch.tensor, mask: torch.tensor) \
            -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        '''
        input:
            + src: input vectors.
            + img: input image.
            + mask: used to mask out padding positions.
        output:
            prediction.
        '''
        tag_vectors = self.tag_dropout(self.tag_embedder(src))
        tag_out = self.tag_encoder(tag_vectors.permute(
            1, 0, 2), src_key_padding_mask=mask)
        tag_out = torch.relu(tag_out.permute(1, 0, 2))

        img_regions = self.img_feature_extractor(img)
        attn_out = self.img_dropout(torch.relu(
            self.tag_to_img(tag_vectors, img_regions)
        ))
        img_out = self.img_encoder(attn_out.permute(
            1, 0, 2), src_key_padding_mask=mask)
        img_out = torch.relu(img_out.permute(1, 0, 2))

        img_weight = self.gating(torch.cat((tag_out, img_out), dim=-1))
        tag_weight = 1 - img_weight

        tag_out = self.tag_sigmoid(self.tag_linear(tag_out).squeeze(dim=-1))
        img_out = self.img_sigmoid(self.img_linear(img_out).squeeze(dim=-1))

        out = tag_weight * tag_out + img_weight * img_out

        return out, tag_out, img_out, tag_weight, img_weight
