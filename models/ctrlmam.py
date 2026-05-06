import torch
import torch.nn as nn
import numpy as np
from openpoints.models.PCM.mamba_layer import MambaBlock
# Hugging Face libraries
from transformers import AutoModel, AutoConfig
import warnings

warnings.filterwarnings("ignore")


# -------------------------- 1. Lightweight PointNet for point-cloud embeddings (LLM dim) --------------------------
class PointNetEmbedding(nn.Module):

    def __init__(self, input_size=3,d_model =1024):
        super().__init__()
        self.input_size = input_size

        self.hideen_size1 = 64
        self.hideen_size4 = 256
        self.hideen_size5 = d_model

        self.act1 = nn.ReLU()

        self.input_norm1 = nn.LayerNorm(self.hideen_size1)
        self.input_norm3 = nn.LayerNorm(self.hideen_size4)
        self.input_norm4 = nn.LayerNorm(self.hideen_size5)
        self.localfc1 = nn.Linear(self.input_size, self.hideen_size1)
        self.globalfc1 = nn.Linear(self.hideen_size1, self.hideen_size4)
        self.globalfc2 = nn.Linear(self.hideen_size4, self.hideen_size5)

        self.conv_k2 = nn.Conv1d(self.hideen_size5, self.hideen_size5, kernel_size=2, stride=1, padding=0)
        self.conv_k3 = nn.Conv1d(self.hideen_size5, self.hideen_size5, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.localfc1(x)
        x = self.input_norm1(x)
        x = self.act1(x)

        x = self.globalfc1(x)
        x = self.input_norm3(x)
        x = self.act1(x)

        x = self.globalfc2(x)
        x = self.input_norm4(x)
        x = self.act1(x)

        cx = x.transpose(1, 2)
        ux = self.act1(self.conv_k2(cx)).transpose(1, 2)
        cx = self.act1(self.conv_k3(cx)).transpose(1, 2)

        return x,ux,cx


# -------------------------- 2. Load frozen Hugging Face pretrained backbone --------------------------
class FrozenHuggingFaceBackbone(nn.Module):
    def __init__(self, model_path, emb_dim=768):
        super().__init__()
        # Load config only (structure); swap in a point-cloud pretrained checkpoint if available
        self.qwen2_config = AutoConfig.from_pretrained('Qwenconfig/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/')
        self.qwen2_model = AutoModel.from_pretrained(
            "Qwenmodel",
            config=self.qwen2_config,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.embedding_layer = self.qwen2_model.embed_tokens

        # Freeze all parameters
        for param in self.qwen2_model.parameters():
            param.requires_grad = False

    def build_attention_mask(self, input_seq_len, output_seq_len, device):
        """
        Build attention mask:
        - Input tokens: attend to all input tokens.
        - Output tokens: causal — attend to all inputs and prior output tokens only.
        """
        total_seq_len = input_seq_len + output_seq_len

        # Base mask: 1 = visible, 0 = masked
        mask = torch.ones((total_seq_len, total_seq_len),dtype=torch.bool, device=device)

        # Causal mask on the output block
        if output_seq_len > 0:
            output_mask = torch.tril(torch.ones((output_seq_len, output_seq_len),dtype=torch.bool, device=device))
            mask[input_seq_len:, input_seq_len:] = output_mask

            # Output cannot attend to input tokens after it (inputs are prefix; N/A here)
            # Input can attend to all output tokens; change if you need bidirectional behavior
            # To hide outputs from input tokens, uncomment:
            # mask[:input_seq_len, input_seq_len:] = 0

        return mask.unsqueeze(0)  # [1, total_seq_len, total_seq_len]

    def forward(self, x,atten):

        #
        # combined_embeds = torch.cat([x, vemb], dim=1).half()

        attention_mask = atten
        # outatten = torch.zeros((x.shape[0],vemb.shape[1]),dtype=torch.long, device=x.device)
        # attention_mask = torch.cat((attention_mask, outatten), dim=1)
        # print(attention_mask)
        outputs = self.qwen2_model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.last_hidden_state.squeeze(1)  # [B, emb_dim]


class kClassificationHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=100):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, 512 ),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.0, 4)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        fused_add_norm = False
        residual_in_fp32 = False,
        mamba_layer_idx = 0
        bimamba_type = 'v1'
        self.mamba_blocks_list = nn.Sequential()
        for n_mamba in range(4):
            mamba_block_module = MambaBlock(dim=512, layer_idx=mamba_layer_idx,
                                            bimamba_type=bimamba_type,
                                            norm_cls=nn.LayerNorm, fused_add_norm=fused_add_norm,
                                            residual_in_fp32=residual_in_fp32,
                                            drop_path=inter_dpr[mamba_layer_idx])
            self.mamba_blocks_list.append(mamba_block_module)
            mamba_layer_idx += 1
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.00)
        self.last = nn.Linear(512, 1)


    def forward(self, x):
        x = self.pre(x)
        for i in range(4):
            x,x_res = self.mamba_blocks_list[i](x)
            x = x+x_res

        x =self.norm(x)
        x= self.dropout(x)
        x = self.last(x)
        return x

# -------------------------- 3. Classification heads --------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=100):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, 512 ),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.0, 8)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        fused_add_norm = False
        residual_in_fp32 = False,
        mamba_layer_idx = 0
        bimamba_type = 'v2'
        self.mamba_blocks_list = nn.Sequential()
        for n_mamba in range(8):
            mamba_block_module = MambaBlock(dim=512, layer_idx=mamba_layer_idx,
                                            bimamba_type=bimamba_type,
                                            norm_cls=nn.LayerNorm, fused_add_norm=fused_add_norm,
                                            residual_in_fp32=residual_in_fp32,
                                            drop_path=inter_dpr[mamba_layer_idx])
            self.mamba_blocks_list.append(mamba_block_module)
            mamba_layer_idx += 1
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.05)
        self.last = nn.Linear(512, 3*num_classes)


    def forward(self, x):
        x = self.pre(x)
        for i in range(8):
            x,x_res = self.mamba_blocks_list[i](x)
            x = x+x_res

        x =self.norm(x)
        x= self.dropout(x)
        x = self.last(x)
        return x


class PointCloudClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased",numclass=1000,droppath = 0.00):
        super().__init__()
        self.emb_dim = 1024  # BERT-base style embedding dim
        self.embedding = PointNetEmbedding(d_model=self.emb_dim)  # Point-cloud embeddings
        self.conv_k3 = nn.Conv1d(self.emb_dim, self.emb_dim, kernel_size=3, stride=3, padding=0)
        dpr = [x.item() for x in torch.linspace(0, droppath, 4)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        fused_add_norm = False
        residual_in_fp32 = False,
        mamba_layer_idx = 0
        bimamba_type = 'v1'
        self.umamba_blocks_list = nn.Sequential()
        for n_mamba in range(4):
            mamba_block_module = MambaBlock(dim=self.emb_dim, layer_idx=mamba_layer_idx,
                                            bimamba_type=bimamba_type,
                                            norm_cls=nn.LayerNorm, fused_add_norm=fused_add_norm,
                                            residual_in_fp32=residual_in_fp32,
                                            drop_path=inter_dpr[mamba_layer_idx])
            self.umamba_blocks_list.append(mamba_block_module)
            mamba_layer_idx += 1

        self.cmamba_blocks_list = nn.Sequential()
        mamba_layer_idx = 0
        for n_mamba in range(4):
            mamba_block_module = MambaBlock(dim=self.emb_dim, layer_idx=mamba_layer_idx,
                                            bimamba_type=bimamba_type,
                                            norm_cls=nn.LayerNorm, fused_add_norm=fused_add_norm,
                                            residual_in_fp32=residual_in_fp32,
                                            drop_path=inter_dpr[mamba_layer_idx])
            self.cmamba_blocks_list.append(mamba_block_module)
            mamba_layer_idx += 1

        self.numclass = numclass

        self.backbone = FrozenHuggingFaceBackbone(  # Frozen HF backbone
            model_path=model_name,
            emb_dim=self.emb_dim
        )
        self.class_head = ClassificationHead(input_dim=self.emb_dim,num_classes=self.numclass)  # Classifier head
        # Knot intervals: 9 segments [4:13], softmax to sum to 1
        self.knot_head = kClassificationHead(input_dim=self.emb_dim,num_classes=self.numclass)

    def forward(self, point,utoken,ctoken,etoken,allaten):


        pemb,uemb,cemb = self.embedding(point)  # [B, N, D] point-cloud embeddings
        B, N, D = uemb.shape
        uemb = uemb.unsqueeze(2)
        cemb = cemb.unsqueeze(2)

        utemb = self.backbone.embedding_layer(utoken).float()
        convutemb = self.conv_k3(utemb.transpose(1, 2)).transpose(1, 2)
        ctemb = self.backbone.embedding_layer(ctoken).float()
        etemb = self.backbone.embedding_layer(etoken)

        convutemb = convutemb.unsqueeze(2)
        ctemb = ctemb.unsqueeze(2)

        uemb = torch.cat([convutemb,uemb], dim=2)

        uemb = uemb.reshape(B,2*N,D)

        cemb = torch.cat([ctemb,cemb], dim=2)
        cemb = cemb.reshape(B,2*(N-1),D)

        uemb,uemb_res = self.umamba_blocks_list[0](uemb)
        cemb,cemb_res = self.cmamba_blocks_list[0](cemb)
        uemb +=uemb_res
        cemb +=cemb_res
        uemb = uemb[:,1::2,:]
        cemb = cemb[:,1::2,:]

        convutemb = convutemb.squeeze(2)
        ctemb = ctemb.squeeze(2)

        emb = torch.cat((pemb.half(),uemb.half(),cemb.half(),etemb,convutemb.half(),ctemb.half()), dim=1)
        feat = self.backbone(emb,allaten)  # [B, L, D] frozen LM hidden states
        pfeat = feat[:,147:159,:].float()
        logits = self.class_head(pfeat)  # Classification logits
        # Knot head: 9 interval segments, softmax rows sum to 1
        knot_feat = feat[:,159:168,:].float()
        seg_logits = self.knot_head(knot_feat)
        seg_logits = seg_logits.reshape(-1,9)   # [B, 9]
        seg = torch.softmax(seg_logits, dim=1)  # Rows sum to 1; only these 9 knot segments
        return logits,seg
