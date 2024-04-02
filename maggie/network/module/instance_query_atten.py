import torch
import torch.nn as nn
from .position_encoding import PositionEmbeddingSine
from .mask_attention import MLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer

class InstanceQueryAttention(nn.Module):
    def __init__(self, input_dim=64, max_inst=10, num_queries=16):
        super().__init__()

        self.input_dim = input_dim
        self.max_inst = max_inst
        self.num_queries = num_queries

        # Initial feature and position embedding (for the first frame)
        self.instance_feat = nn.Embedding(num_queries, input_dim)
        self.instance_pos = nn.Embedding(num_queries, input_dim)
        nn.init.xavier_uniform_(self.instance_feat.weight)
        nn.init.xavier_uniform_(self.instance_pos.weight)

        self.pe_embedding = PositionEmbeddingSine(input_dim//2)

        # Attention block
        self.feat_proj = FFNLayer(d_model=input_dim, dim_feedforward=input_dim, normalize_before=False) #MLP(input_dim, input_dim, input_dim, 1)
        self.sa_block = SelfAttentionLayer(d_model=input_dim, nhead=1, normalize_before=False)
        self.instance_feat_ca_block = CrossAttentionLayer(d_model=input_dim, nhead=1, normalize_before=False)
        self.ffn_block = FFNLayer(d_model=input_dim, dim_feedforward=input_dim, normalize_before=False)
        self.feat_instance_ca_block = CrossAttentionLayer(d_model=input_dim, nhead=1, normalize_before=False)
        self.feat_mlp = FFNLayer(d_model=input_dim, dim_feedforward=input_dim, normalize_before=False)
        self.instance_mlp = FFNLayer(d_model=input_dim, dim_feedforward=input_dim, normalize_before=False)

        # LSTM to keep track instance queries
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, n_i, previous_mem=None):
        '''
        x: SparseTensor (batch_size, max_inst, input_dim)
        '''
        indices = x.indices
        features = x.features
        h, w = x.spatial_shape
        batch_size = x.batch_size // n_i
        
        # Compute batch indices and instance indices
        batch_indices = torch.div(indices[:, 0], n_i, rounding_mode='trunc').long()
        inst_indices = (indices[:, 0] % n_i).long()

        # Feature projection
        features = self.feat_proj(features)

        # x = x.replace_feature(features)
        # return x, None

        # Compute feat position embedding
        feature_pos = self.pe_embedding(torch.zeros(batch_size, self.input_dim, h, w, device=features.device))

        feature_pos = feature_pos[batch_indices, :, indices[:, 1].long(), indices[:, 2].long()]

        # Prepare instance queries
        if previous_mem is None:
            inst_queries = self.instance_feat.weight
            inst_queries = inst_queries[None, :, None].repeat(self.max_inst, 1, batch_size, 1).flatten(0, 1)
        else:
            # import pdb; pdb.set_trace()
            inst_queries = previous_mem[0]
        inst_pos = self.instance_pos.weight
        inst_pos = inst_pos[None, :, None].repeat(self.max_inst, 1, batch_size, 1).flatten(0, 1)

        # Prepare padding mask
        # 1: ignore, 0: accept
        mask = torch.ones((indices.shape[0], self.max_inst, batch_size), dtype=torch.bool, device=inst_queries.device)
        mask[torch.arange(indices.shape[0]), inst_indices, batch_indices] = False
        mask = mask[:, :, None].repeat(1, 1, self.num_queries, 1)
        mask = mask.flatten(1,3).permute(1,0)

        # Self-attention between instance queries
        # import pdb; pdb.set_trace()
        # inst_queries = self.sa_block(inst_queries,  tgt_mask=None,  tgt_key_padding_mask=None, query_pos=inst_pos)

        # Convert tokens to N, input_dim
        inst_queries = inst_queries.flatten(0, 1)
        inst_pos = inst_pos.flatten(0, 1)


        # Query memory if there exists the previous memory
        if previous_mem is not None:
            features, _ = self.feat_instance_ca_block(features[:, None], inst_queries[:, None], memory_mask=mask.transpose(0,1)[None], query_pos=feature_pos[:, None], pos=inst_pos[:, None])
            features = features.squeeze(1)
            features = self.feat_mlp(features)

        x = x.replace_feature(features)

        # Update memory by querying current features
        inst_queries, _ = self.instance_feat_ca_block(inst_queries[:, None], features[:, None], memory_mask=mask[None], pos=feature_pos[:, None], query_pos=inst_pos[:, None])
        # inst_queries = inst_queries.squeeze(1).reshape(self.max_inst * self.num_queries, batch_size, self.input_dim)
        inst_queries = torch.nan_to_num(inst_queries, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM to update memory
        inst_queries = inst_queries.squeeze(1).reshape(self.max_inst * self.num_queries, batch_size, self.input_dim)
        if previous_mem is None:
            inst_queries, hidden_state = self.lstm(inst_queries.flatten(0,1)[:, None])
        else:
            inst_queries, hidden_state = self.lstm(inst_queries.flatten(0,1)[:, None], previous_mem[1])
        inst_queries = inst_queries.squeeze(1).reshape(-1, batch_size, self.input_dim)
        inst_queries = self.instance_mlp(inst_queries)

        return x, (inst_queries, hidden_state)

        # Cross attention between instance queries and instance features
        # import pdb; pdb.set_trace()
        # inst_queries, _ = self.instance_feat_ca_block(inst_queries[:, None], features[:, None], memory_mask=mask[None], pos=feature_pos[:, None], query_pos=inst_pos[:, None])
        # inst_queries = inst_queries.squeeze(1).reshape(self.max_inst, self.num_queries, batch_size, self.input_dim)
        # inst_queries = torch.nan_to_num(inst_queries, nan=0.0, posinf=0.0, neginf=0.0)
        # inst_queries = self.ffn_block(inst_queries)

        

        # TODO: Can we move the memory here?
         
        # Cross attention between instance features and instance queries
        # features, _ = self.feat_instance_ca_block(features[:, None], inst_queries, memory_mask=mask.transpose(0,1)[None], query_pos=feature_pos[:, None], pos=inst_pos[:, None])
        # features = features.squeeze(1)

        # FFN for features
        # features = self.feat_mlp(features)

        # Update with old memory using LSTM
        # inst_queries = inst_queries.squeeze(1).reshape(self.max_inst * self.num_queries, batch_size, self.input_dim)
        # if previous_mem is None:
        #     inst_queries, hidden_state = self.lstm(inst_queries.flatten(0,1)[:, None])
        # else:
        #     inst_queries, hidden_state = self.lstm(inst_queries.flatten(0,1)[:, None], previous_mem[1])
        # inst_queries = inst_queries.squeeze(1).reshape(-1, batch_size, self.input_dim)
        # inst_queries = self.instance_mlp(inst_queries)

        # Replace with new features
        # x = x.replace_feature(features)
        
        # import pdb; pdb.set_trace()
        # return x, (inst_queries, hidden_state)
        # return x, None





