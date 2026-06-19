import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import init

from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.bert import BertConfig, BertForMaskedLM

from utils import read_json
import einops

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            mask_attn=False,
    ):
        super().__init__()
        self.mask_attn = mask_attn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wkv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        
        q = self.wq(x)
        q = einops.rearrange(q, 'B N (H D) -> B H N D', H=self.num_heads)
        
        kv = self.wkv(cond) # BMD
        kv = einops.rearrange(kv, 'B N (K H D) ->K B H N D', H=self.num_heads, K=2)
        k = kv[0]
        v = kv[1]


        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = einops.rearrange(x, 'B H N D -> B N (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply


def build_vision_encoder(config):

    vision_config = read_json(config['vision_config'])
    vision_width = vision_config['vision_width']

    vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                     h=vision_config['h'],
                                     w=vision_config['w'],
                                     embed_dim=vision_config['embed_dim'],
                                     depths=vision_config['depths'],
                                     num_heads=vision_config['num_heads'],
                                     window_size=vision_config['window_size'],
                                     drop_rate=vision_config['drop_rate'],
                                     drop_path_rate=vision_config['drop_path_rate'])

    if config['load_params']:
        # download from https://github.com/microsoft/Swin-Transformer
        state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']
        print(f"build_vision_encoder: load swin {vision_config['h']} x {vision_config['w']} ====>")
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys, flush=True)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width):

    config_text = BertConfig.from_json_file(config['text_config'])
    config_text.encoder_width = vision_width
    text_encoder, msg = BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                        output_loading_info=True)
    if config['load_params']:
        print("build_text_encoder: load bert ====>")
        for k, v in msg.items():
            print(f"{k}: {sorted(v)}")

    return text_encoder


def feature_mapping(input_dim, output_dim, dropout_p=0):
    mlp = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(input_dim, output_dim),
    )
    init.normal_(mlp[2].weight.data, std=0.00001)
    init.constant_(mlp[2].bias.data, 0.0)
    return mlp

class CMP(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # image encoder
        self.vision_encoder, vision_width = build_vision_encoder(config)
        self.vision_width = vision_width

        # text & cross encoder
        self.text_encoder = build_text_encoder(config, vision_width=self.vision_width)
        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width

        self.embed_dim = config['embed_dim']
        self.avgpool = nn.AdaptiveAvgPool1d(3)
        self.vision_proj = feature_mapping(1024*3, self.embed_dim, config['itc_dp'])
        self.text_proj = feature_mapping(3840, self.embed_dim, config['itc_dp'])
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        self.epsilon = config['label_smooth']
        print('Label Smooth:', self.epsilon)

        input_dim = self.text_width
        output_dim = 2
        self.itm_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim))

        self.init_params = []  # train with 2 * lr
        # image encoder
        for i in range(2, 4):
            for name, param in self.vision_encoder.layers[i].named_parameters():
                self.init_params.extend(['vision_encoder.layers.' + str(i) + '.' + name])

        # text & cross encoder
        temp_encoder = self.text_encoder.bert
        temp_name = 'text_encoder.bert.encoder.layer.'
        temp_list = [4, 5, 10, 11]
        for i in temp_list:
            for name, param in temp_encoder.encoder.layer[i].named_parameters():
                self.init_params.extend([temp_name + str(i) + '.' + name])
        self.init_params.extend(
            ['text_encoder.cls.' + n for n, _ in self.text_encoder.cls.named_parameters()])

        self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
        self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])
        self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])
        self.decoupled_linear_img = torch.nn.Linear(1024, 1024*3)
        self.decoupled_linear_txt = torch.nn.Linear(768, 768*3)


        self.init_params.extend(['decoupled_linear_img.' + n for n, _ in self.decoupled_linear_img.named_parameters()])
        self.init_params.extend(['decoupled_linear_txt.' + n for n, _ in self.decoupled_linear_txt.named_parameters()])
        
        self.use_ca = config.get("use_ca", False)
        if self.use_ca:
            self.cross_attn_img = CrossAttention(dim=1024)
            self.cross_attn_txt = CrossAttention(dim=768)
        
        self.modality_proj_img = torch.nn.Linear(1024*2, 1024)
        self.modality_proj_txt = torch.nn.Linear(768*2, 1024)
        self.init_params.extend(['modality_proj_img.' + n for n, _ in self.modality_proj_img.named_parameters()])
        self.init_params.extend(['modality_proj_txt.' + n for n, _ in self.modality_proj_txt.named_parameters()])




    def load_pretrained(self, ckpt_rpath):
        checkpoint = torch.load(ckpt_rpath, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        msg = self.load_state_dict(state_dict, strict=False, assign=True)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys])
        print("unexpected_keys: ", msg.unexpected_keys)


    def get_vision_embeds(self, image):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts


    def get_text_embeds(self, text_ids, text_atts):
        encoder = self.text_encoder.bert
        return encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text').last_hidden_state


    def get_cross_embeds(self, image_embeds, image_atts, text_embeds, text_atts,):
        encoder = self.text_encoder.bert
        return encoder(encoder_embeds=text_embeds,
                       attention_mask=text_atts,
                       encoder_hidden_states=image_embeds,
                       encoder_attention_mask=image_atts,
                       return_dict=True,
                       mode='fusion',
                       ).last_hidden_state

    def get_image_feat(self, image_embeds):
        image_feat, _, _, _ = self.get_image_feat_decoupled(image_embeds)
        return image_feat


    def get_image_feat_decoupled(self, image_embeds):
        x_cls = image_embeds[:, 0, :]
        x_embeds = image_embeds[:, 1:, ]

      
        image_app_feat, image_act_feat, image_bg_feat = self.decoupled_linear_img(x_cls).chunk(3, dim=-1)
        if self.use_ca:
            q = torch.stack([image_app_feat, image_act_feat, image_bg_feat], dim=1)  # B, 3, D
            weighted_app_feat, weighted_act_feat, weighted_bg_feat = self.cross_attn_img(q, x_embeds).chunk(3, dim=1)

            weighted_app_feat = weighted_app_feat.squeeze(1)
            weighted_act_feat = weighted_act_feat.squeeze(1)
            weighted_bg_feat = weighted_bg_feat.squeeze(1)
        else:
            x_app_sim = F.sigmoid(torch.matmul(x_embeds, image_app_feat.unsqueeze(-1)).squeeze(-1))
            x_act_sim = F.sigmoid(torch.matmul(x_embeds, image_act_feat.unsqueeze(-1)).squeeze(-1))
            x_bg_sim = F.sigmoid(torch.matmul(x_embeds, image_bg_feat.unsqueeze(-1)).squeeze(-1))

            weighted_app_feat = torch.sum(x_embeds * x_app_sim.unsqueeze(-1), dim=1)
            weighted_act_feat = torch.sum(x_embeds * x_act_sim.unsqueeze(-1), dim=1)
            weighted_bg_feat =  torch.sum(x_embeds * x_bg_sim.unsqueeze(-1), dim=1)

            
        image_app_feat = self.modality_proj_img(torch.cat([image_app_feat, weighted_app_feat], dim=1))
        image_act_feat = self.modality_proj_img(torch.cat([image_act_feat, weighted_act_feat], dim=1))
        image_bg_feat = self.modality_proj_img(torch.cat([image_bg_feat, weighted_bg_feat], dim=1))


        x = torch.cat([image_app_feat, image_act_feat, image_bg_feat], dim=-1)
        image_feat = self.vision_proj(x)

        return image_feat, image_app_feat, image_act_feat, image_bg_feat

    def get_image_feat_decoupled_visual(self, image_embeds):
        x_cls = image_embeds[:, 0, :]
        x_embeds = image_embeds[:, 1:, ]

        image_app_feat, image_act_feat, image_bg_feat = self.decoupled_linear_img(x_cls).chunk(3, dim=-1)

        x_app_sim = F.sigmoid(torch.matmul(x_embeds, image_app_feat.unsqueeze(-1)).squeeze(-1))
        x_act_sim = F.sigmoid(torch.matmul(x_embeds, image_act_feat.unsqueeze(-1)).squeeze(-1))
        x_bg_sim = F.sigmoid(torch.matmul(x_embeds, image_bg_feat.unsqueeze(-1)).squeeze(-1))

        return x_app_sim, x_act_sim, x_bg_sim



    def get_text_feat_decoupled(self, text_embeds):
        x_cls = text_embeds[:, 0, :]
        x_embeds = text_embeds[:, 1:, ]

        text_app_feat, text_act_feat, text_bg_feat = self.decoupled_linear_txt(x_cls).chunk(3, dim=-1)
        if self.use_ca:
            q = torch.stack([text_app_feat, text_act_feat, text_bg_feat], dim=1)  # B, 3, D
            weighted_app_feat, weighted_act_feat, weighted_bg_feat = self.cross_attn_txt(q, x_embeds).chunk(3, dim=1)
            
            weighted_app_feat = weighted_app_feat.squeeze(1)
            weighted_act_feat = weighted_act_feat.squeeze(1)
            weighted_bg_feat = weighted_bg_feat.squeeze(1)
        else:
            x_app_sim = F.sigmoid(torch.matmul(x_embeds, text_app_feat.unsqueeze(-1)).squeeze(-1))
            x_act_sim = F.sigmoid(torch.matmul(x_embeds, text_act_feat.unsqueeze(-1)).squeeze(-1))
            x_bg_sim = F.sigmoid(torch.matmul(x_embeds, text_bg_feat.unsqueeze(-1)).squeeze(-1))
                                                                    
            weighted_app_feat = torch.sum(x_embeds * x_app_sim.unsqueeze(-1), dim=1)
            weighted_act_feat = torch.sum(x_embeds * x_act_sim.unsqueeze(-1), dim=1)
            weighted_bg_feat =  torch.sum(x_embeds * x_bg_sim.unsqueeze(-1), dim=1)

        text_app_feat = self.modality_proj_txt(torch.cat([text_app_feat, weighted_app_feat], dim=1))
        text_act_feat = self.modality_proj_txt(torch.cat([text_act_feat, weighted_act_feat], dim=1))
        text_bg_feat = self.modality_proj_txt(torch.cat([text_bg_feat, weighted_bg_feat], dim=1))
    
        x = torch.cat([x_cls, text_app_feat, text_act_feat, text_bg_feat], dim=-1)
        text_feat = self.text_proj(x)

        return text_feat, text_app_feat, text_act_feat, text_bg_feat

    def get_text_feat_decoupled_visual(self, text_embeds):
        x_cls = text_embeds[:, 0, :]
        x_embeds = text_embeds[:, 1:, ]

        text_app_feat, text_act_feat, text_bg_feat = self.decoupled_linear_txt(x_cls).chunk(3, dim=-1)

        x_app_sim = F.sigmoid(torch.matmul(x_embeds, text_app_feat.unsqueeze(-1)).squeeze(-1))
        x_act_sim = F.sigmoid(torch.matmul(x_embeds, text_act_feat.unsqueeze(-1)).squeeze(-1))
        x_bg_sim = F.sigmoid(torch.matmul(x_embeds, text_bg_feat.unsqueeze(-1)).squeeze(-1))
                                        
        return x_app_sim, x_act_sim, x_bg_sim

    def get_text_feat(self, text_embeds):
        text_feat, _, _, _ = self.get_text_feat_decoupled(text_embeds)
        return text_feat


    def get_contrastive_loss(self, image_feat, text_feat, idx):
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        idx = idx.view(-1, 1)
        assert idx.size(0) == image_feat.size(0)
        idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)

        if 0 < self.epsilon < 1:
            _, num_classes = logits.shape
            labels = (1 - self.epsilon) * labels + self.epsilon / num_classes

        loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2


    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None, ):
        """
        Matching Loss with in-batch hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            idx = idx.view(-1, 1)
            assert idx.size(0) == bs
            mask = torch.eq(idx, idx.t())
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts,
                                          text_embeds=text_embeds, text_atts=text_atts,)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all,
                                          text_embeds=text_embeds_all, text_atts=text_atts_all,)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        return itm_loss


    def get_matching_loss_hard(self, image_embeds, image_atts, image_embeds_hard, image_atts_hard,
                               text_embeds, text_atts, text_embeds_hard, text_atts_hard):
        bs = image_embeds.size(0)
        image_embeds = torch.cat([image_embeds, image_embeds_hard], dim=0)
        image_atts = torch.cat([image_atts, image_atts_hard], dim=0)
        text_embeds = torch.cat([text_embeds_hard, text_embeds], dim=0)
        text_atts = torch.cat([text_atts_hard, text_atts], dim=0)

        cross_hard = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds,
                                          text_atts=text_atts)[:, 0, :]
        output = self.itm_head(cross_hard)

        # Matching Loss with only hard negatives for pab
        itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long),
                                torch.zeros(bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)
        return itm_loss

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids, ):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss
    

    def get_matching_loss_decoupled(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, text_dec_embeds, text_dec_att, idx=None, ):
        """
        Matching Loss with in-batch hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            idx = idx.view(-1, 1)
            assert idx.size(0) == bs
            mask = torch.eq(idx, idx.t())
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts,
                                          text_embeds=text_dec_embeds, text_atts=text_dec_att,)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all,
                                          text_embeds=text_embeds_all, text_atts=text_atts_all,)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        return itm_loss
