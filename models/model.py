from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.head import Projector
from datasets.templates import get_templates
from tools.model import load_checkpoint
from solver import build_optimizer, build_scheduler_iter
from .base import BaseModel
from .bonder import CrossAttnBlock

import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os.path as osp


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # Basic config
        n_cls = len(classnames)
        ctx_init = cfg.TRAINER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vocab_size = clip_model.vocab_size
        transformer_width = clip_model.transformer_width
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        _tokenizer = _Tokenizer()
        device = 'cuda'

        # Prompt Queries
        num_q_category = cfg.MODEL.BONDER.NUM_Q_CTG
        num_q_content = cfg.MODEL.BONDER.NUM_Q_CON
        self.query_category = nn.Parameter(torch.zeros(1, num_q_category, ctx_dim))
        self.query_content = nn.Parameter(torch.zeros(1, num_q_content, ctx_dim))
        self.query_category.data.normal_(mean=0.0, std=0.02)
        self.query_content.data.normal_(mean=0.0, std=0.02)

        # Bonder
        self.bonder_category = CrossAttnBlock(ctx_dim, num_heads=8)
        self.bonder_content = CrossAttnBlock(ctx_dim, num_heads=8)
        print(f"Number of category-wise queries: {num_q_category}")
        print(f"Number of content-wise queries: {num_q_content}")

        # Vocab Head
        self.vocab_head = nn.Linear(transformer_width, vocab_size, bias=False)

        # Initialize prompt queries
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        self.query_category.data[:, :n_ctx] = ctx_vectors[:, :]
        print(f"Initialize category query with: {ctx_init}")
        print(f"Random initialization for content query")

        if cfg.TRAINER.PREC == "fp16":
            self.bonder_category.half()
            self.bonder_content.half()

        # Pseudo sentence
        classnames = [name.replace("_", " ").replace("(", " ").replace(")", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        pseudo_query_category = ["X" + " X" * (num_q_category - 1) + '.']
        pseudo_query_content = ["X" + " X" * (num_q_content - 1) + '.']
        label_sentence = [f"a photo of a {n}." for n in classnames]

        tokenized_query_category = clip.tokenize(pseudo_query_category)
        tokenized_query_content = clip.tokenize(pseudo_query_content)
        tokenized_label = clip.tokenize(label_sentence)
        tokenized_label = tokenized_label[:, 1:1 + num_q_category]
        with torch.no_grad():
            embedding_category = clip_model.token_embedding(tokenized_query_category).type(dtype)
            embedding_content = clip_model.token_embedding(tokenized_query_content).type(dtype)

        # Save but not trainable
        self.register_buffer("token_prefix_cat", embedding_category[:, :1, :])
        self.register_buffer("token_suffix_cat", embedding_category[:, 1 + num_q_category:, :])
        self.register_buffer("token_prefix_con", embedding_content[:, :1, :])
        self.register_buffer("token_suffix_con", embedding_content[:, 1 + num_q_content:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.device = device
        self.tokenized_query_category = tokenized_query_category.to(device)
        self.tokenized_query_content = tokenized_query_content.to(device)
        self.label = tokenized_label.masked_fill(tokenized_label == 0, -100).to(device)
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_q_content
        self.vocab_size = vocab_size
        
        # Loss for text supervision, better with label smoothing
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    def forward(self, im_features, im_cls, target=None, caption=None):
        prefix_cat = self.token_prefix_cat.expand(im_features.size(0), -1, -1)
        suffix_cat = self.token_suffix_cat.expand(im_features.size(0), -1, -1)
        prefix_con = self.token_prefix_con.expand(im_features.size(0), -1, -1)
        suffix_con = self.token_suffix_con.expand(im_features.size(0), -1, -1)

        # category-wise
        query_category = self.query_category.expand(im_features.size(0), -1, -1).clone()
        query_category.data[:, self.n_ctx] = im_cls[:, :]
        query_output_category = self.bonder_category(self.query_category, im_features)  # [B, num_q, dim]

        # content-wise
        query_content = self.query_content.expand(im_features.size(0), -1, -1).clone()
        query_content.data[:, self.n_ctx] = im_cls[:, :]
        query_output_content = self.bonder_content(self.query_content, im_features)  # [B, num_q, dim]

        if self.training:
            # category-wise
            query_output_vocab_category = self.vocab_head(query_output_category)
            category_target = self.label[target]  # [B,num_q]
            loss_category = self.criterion(
                query_output_vocab_category.view(-1, self.vocab_size),
                category_target.view(-1)
            )

            # content-wise
            query_output_vocab_content = self.vocab_head(query_output_content)
            caption = caption[:, 1:1 + self.num_query_token]    # [B, num_q]
            content_target = caption.masked_fill(caption == 0, -100).to(self.device)
            loss_content = self.criterion(
                query_output_vocab_content.view(-1, self.vocab_size),
                content_target.view(-1)
            )
        else:
            # dummy for inference stage
            loss_category, loss_content = 0, 0

        prompts_category = torch.cat([prefix_cat, query_output_category, suffix_cat], dim=1)  # [B, 77, dim]
        prompts_content = torch.cat([prefix_con, query_output_content, suffix_con], dim=1)   # [B, 77, dim]

        return prompts_category, prompts_content, loss_category, loss_content


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        # Projector
        self.cls_head = Projector(classnames, clip_model, self.logit_scale)
        self.wiseft_head = Projector(classnames, clip_model, self.logit_scale)  # WISEFT, not used

        # shared weights
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        # zero-shot weights
        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zs_weights = self.get_zero_shot_weights(classnames, clip_model).to(device)
        self.cls_head.fc.weight.data = self.zs_weights.clone()

    def get_zero_shot_weights(self, classnames, clip_model, device="cuda"):
        num_classes = len(classnames)
        self.text_encoder.to(device)
        with torch.no_grad():
            weights = torch.empty_like(self.cls_head.fc.weight.data)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in self.text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).type(self.dtype)
                text_embedding = text_embedding.to(device)

                text_features = self.text_encoder(text_embedding, text_tokenized)
                # average across all templates
                text_features = text_features.mean(dim=0)
                text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        return weights

    def forward(self, image, target=None, caption=None):
        tokenized_prompts_category = self.prompt_learner.tokenized_query_category
        tokenized_prompts_content = self.prompt_learner.tokenized_query_content

        # image feature
        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        image_cls = image_cls.float()
        
        # two prompts: prompts_category, prompts_content
        # two loss: loss_category, loss_content
        prompts_category, prompts_content, loss_category, loss_content = self.prompt_learner(image_features, image_cls, target, caption)

        # text feature
        text_features_category = self.text_encoder(prompts_category, tokenized_prompts_category)
        text_features_content = self.text_encoder(prompts_content, tokenized_prompts_content)
        text_features_category, text_features_content = text_features_category.float(), text_features_content.float()

        # fuse text feature over embedding space
        text_features = (text_features_category + text_features_content) / 2.

        # concat image feature and text feature
        fused_fea = torch.cat([image_cls, text_features], dim=1)
        logits = self.cls_head(fused_fea)
        
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            return logits, logits_wiseft

        return logits, loss_category, loss_content


class TGPT_Model(BaseModel):
    def __init__(self, cfg, classnames=None):
        super().__init__()
        self.logger = logging.getLogger(cfg.TRAINER.NAME)
        self.check_cfg(cfg)
        self.cfg = cfg
        self.test_freq = cfg.TRAIN.TEST_FREQ

        self.logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            clip_model.float()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.logger.info("Building TGPT_Model")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Fix the weights of vocab head
        self.model.prompt_learner.vocab_head.weight.requires_grad_(False)

        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None, caption=None):
        return self.model(image, label, caption)     # logits

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        flags = torch.zeros(len(names))
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"
        
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for i,name in enumerate(names):
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            
            flags[i] = 1
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

        return flags.sum() == len(flags)