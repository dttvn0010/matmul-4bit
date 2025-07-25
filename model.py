import gc
import torch
from torch import nn
from dataclasses import dataclass
import glob
from typing import Optional, Tuple
from ctypes import  CDLL, c_void_p, c_int

lib = CDLL('matmul_q4.so')
lib.populate_weight.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int]
lib.populate_weight.restype = None

lib.matmul_q4.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int]
lib.matmul_q4.restype = None

@dataclass
class QwenConfig:
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2560
    intermediate_size: int = 9728
    max_position_embeddings: int = 40960
    num_attention_heads: int = 32
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    vocab_size: int = 151936
    awq_group_size:int = 128
    use_bf_scales:bool= False
    version:str = "2"
    
shifts = [ 0, 16,  4, 20,  8, 24, 12, 28]

def cvt_qweights(qweights):
    n_inp = qweights.shape[0]
    n_out = qweights.shape[1] * 8

    qweights_i8 = torch.zeros((n_inp, n_out), dtype=torch.uint8, device=qweights.device)

    for k in range(8):
        qweights_i8[:, k::8] = ((qweights >> shifts[k]) & 0xF).type(torch.uint8)

    qweights_cvt = torch.zeros((n_inp//8, n_out), dtype=torch.int32, device=qweights.device)
    qweights_cvt |= (qweights_i8[0::8])
    qweights_cvt |= (qweights_i8[1::8].to(torch.int32) << 4)
    qweights_cvt |= (qweights_i8[2::8].to(torch.int32) << 8)
    qweights_cvt |= (qweights_i8[3::8].to(torch.int32) << 12)
    qweights_cvt |= (qweights_i8[4::8].to(torch.int32) << 16)
    qweights_cvt |= (qweights_i8[5::8].to(torch.int32) << 20)
    qweights_cvt |= (qweights_i8[6::8].to(torch.int32) << 24)
    qweights_cvt |= (qweights_i8[7::8].to(torch.int32) << 28)

    return qweights_cvt # (n_inp//8, n_out)

def cvt_qzeros(qzeros):
    n_inp_group = qzeros.shape[0]
    n_out = qzeros.shape[1] * 8
    zeros = torch.zeros((n_inp_group, n_out), dtype=torch.uint8, device=qzeros.device)
    for k in range(8):
        zeros[:,k::8] = ((qzeros >> shifts[k]) & 0xF).type(torch.bfloat16)
    return zeros  # (n_inp//awq_group_size, n_out)


def populate_weight(qweight, scales, qzeros, tranpose=False):
    n_inp = qweight.shape[0] * 8
    n_out = qweight.shape[1]

    weight = torch.zeros(
        (n_out, n_inp) if tranpose 
        else (n_inp, n_out), 
        device=qweight.device, 
        dtype=torch.bfloat16
    )

    awq_group_size = n_inp// scales.shape[0]
    lib.populate_weight(
        qweight.data_ptr(), 
        scales.data_ptr(), 
        qzeros.data_ptr(), 
        weight.data_ptr(), 
        n_inp, 
        n_out, 
        awq_group_size,
        1 if tranpose else 0,
        1 if scales.dtype == torch.bfloat16 else 0,
    )
    return weight

def matmul_q4(inp, qweight, scales, qzeros):
    n_inp = qweight.shape[0] * 8
    n_out = qweight.shape[1]
    input_shape = inp.shape
    inp = inp.contiguous()
    awq_group_size = n_inp// scales.shape[0]

    if len(input_shape) > 2:
        inp = inp.reshape(-1, input_shape[-1])

    batch = inp.shape[0]

    if batch > 32:
        weight = populate_weight(qweight, scales, qzeros, True)
        output = nn.functional.linear(inp, weight)
    else:
        output = torch.zeros(batch, n_out, device=inp.device, dtype=torch.float)

        lib.matmul_q4(
            inp.data_ptr(),
            qweight.data_ptr(), 
            scales.data_ptr(), 
            qzeros.data_ptr(), 
            output.data_ptr(), 
            batch,
            n_inp, 
            n_out,
            awq_group_size,
            1 if scales.dtype == torch.bfloat16 else 0,
        )

    if len(input_shape) > 2:
        output = output.reshape(*input_shape[:-1], n_out)

    return output.bfloat16()

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        awq_group_size:int=128,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.qweight = nn.Parameter(
            torch.empty((in_features//8, out_features),  dtype=torch.uint32, device=device),
            requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.empty((in_features//awq_group_size, out_features), dtype=torch.bfloat16, device=device),
            requires_grad=False
        )
        self.qzeros = nn.Parameter(
            torch.empty((in_features//awq_group_size, out_features), dtype=torch.uint8, device=device),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            nn.init.normal_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = matmul_q4(x, self.qweight, self.scales, self.qzeros)
        return y + self.bias if self.bias is not None else y
    
class QwenRMSNorm(nn.Module):
    def __init__(self, config: QwenConfig, hidden_size=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size or config.hidden_size))
        self.eps = config.rms_norm_eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_dtype)
        return x

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_causal_mask(q_len:int, k_len:int, start_pos: torch.IntTensor|int, device: torch.device):
    if isinstance(start_pos, int):
        assert (k_len - start_pos - q_len) >= 0
        tri_matrix = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool, device=device))
        left_pad_matrix = torch.ones(q_len, start_pos, dtype=torch.bool, device=device)
        right_pad_maxtrix = torch.zeros(q_len, k_len - start_pos - q_len, dtype=torch.bool, device=device)
        mask = torch.cat((left_pad_matrix, tri_matrix, right_pad_maxtrix), dim=1)
        mask = mask[None, None]
    else:
        k_range = torch.arange(k_len, dtype=torch.int64, device=device)
        q_range = torch.arange(q_len, dtype=torch.int64, device=device)
        mask = k_range[None, None, None] < q_range[None,None,:,None] + start_pos[:,None,None,None] + 1

    return mask.contiguous()

class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim #self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        qkv_bias = config.version[0] == "2"
        factory_kwargs = {
            "awq_group_size": config.awq_group_size
        }

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias, **factory_kwargs)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias, **factory_kwargs)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias, **factory_kwargs)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, **factory_kwargs)

        if not qkv_bias:
            self.q_norm = QwenRMSNorm(config, self.head_dim)
            self.k_norm = QwenRMSNorm(config, self.head_dim)
        else: 
            self.q_norm = None
            self.k_norm = None


    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ):
        cache_shape = (max_batch_size, max_seq_len, self.num_key_value_heads, self.head_dim)
        key_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
        value_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
        self.register_buffer("key_cache", key_cache, persistent=False)
        self.register_buffer("value_cache", value_cache, persistent=False)

    def del_kv_cache(self):
        self.key_cache = None
        self.value_cache = None

    @torch.no_grad
    def infer(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[torch.LongTensor|int],
        item_indexes : Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xq = self.q_norm(xq) if self.q_norm else xq
        xk = xk.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        xk = self.k_norm(xk) if self.k_norm else xk
        xv = xv.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        cos_emb, sin_emb = pos_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos_emb, sin_emb, unsqueeze_dim=2)
        attn_mask = None
        
        start_pos = start_pos if start_pos is not None else 0

        if isinstance(start_pos, int):
            end_pos = start_pos + seqlen
            if item_indexes is not None:
                self.key_cache[item_indexes, start_pos:end_pos] = xk
                self.value_cache[item_indexes, start_pos:end_pos] = xv
                xk = self.key_cache[item_indexes, :end_pos]
                xv = self.value_cache[item_indexes, :end_pos]
            else:
                self.key_cache[:bsz, start_pos:end_pos] = xk
                self.value_cache[:bsz, start_pos:end_pos] = xv
                xk = self.key_cache[:bsz, :end_pos]
                xv = self.value_cache[:bsz, :end_pos]
        else:
            end_pos = start_pos.max() + seqlen
            if item_indexes is not None:
                batch_idx = item_indexes.repeat_interleave(seqlen)
            else:
                batch_idx = torch.arange(bsz, device=x.device).repeat_interleave(seqlen)
            
            seq_idx = torch.arange(seqlen, dtype=torch.int64, device=x.device)[None].repeat(bsz, 1) + start_pos[:,None]
            seq_idx = seq_idx.flatten()

            # Update cache
            self.key_cache[batch_idx, seq_idx] = xk.view(-1, self.num_key_value_heads, self.head_dim)
            self.value_cache[batch_idx, seq_idx] = xv.view(-1, self.num_key_value_heads, self.head_dim)
            if item_indexes is not None:
                xk = self.key_cache[item_indexes, :end_pos.item()]
                xv = self.value_cache[item_indexes, :end_pos.item()]
            else:
                xk = self.key_cache[:bsz, :end_pos]
                xv = self.value_cache[:bsz, :end_pos]

        if end_pos > seqlen:
            attn_mask = get_causal_mask(seqlen, xk.shape[1], start_pos, x.device)

        kwargs = {} if attn_mask is not None else {"is_causal": True}
        
        output = torch.nn.functional.scaled_dot_product_attention(
            query=xq.transpose(1, 2).bfloat16(),
            key=xk.transpose(1, 2).bfloat16(),
            value=xv.transpose(1, 2).bfloat16(),
            attn_mask=attn_mask,
            enable_gqa=True,
            **kwargs
        ).transpose(1, 2)

        output = output.view(bsz, seqlen, -1)
        return self.o_proj(output)

class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        factory_kwargs = {
            "awq_group_size": config.awq_group_size
        }

        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False, **factory_kwargs)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False, **factory_kwargs)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False, **factory_kwargs)

    def forward(self, x):
        dtype = x.dtype
        x = x.bfloat16()
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)).to(dtype)

class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.self_attn = QwenAttention(config, layer_idx)

        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config)
        self.post_attention_layernorm = QwenRMSNorm(config)

    @torch.no_grad
    def infer(
        self,
        x: torch.Tensor,
        pos_emb: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[torch.LongTensor|int],
        item_indexes: Optional[torch.LongTensor]=None,
    ) -> torch.FloatTensor:
        x = x + self.self_attn.infer(self.input_layernorm(x), pos_emb, start_pos, item_indexes)
        return x + self.mlp(self.post_attention_layernorm(x))

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: QwenConfig, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        dim = config.head_dim
        with torch.autocast(device_type=device, dtype=torch.float32):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, pos, dtype):
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        pos = pos[:, None, :].float()
        with torch.autocast(device_type=pos.device.type, enabled=False):
            freqs = (inv_freq.float().to(pos.device) @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

class QwenModel(nn.Module):

    def __init__(self, config: QwenConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.eos_token_id = config.eos_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size,
        )

        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config, device)

        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = QwenRMSNorm(config)

        if not config.tie_word_embeddings:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.FloatTensor,
        start_pos: Optional[torch.LongTensor|int]=None,
        item_indexes: Optional[torch.LongTensor]=None,
    ):
        bs = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = position_ids[None].repeat(bs, 1)

        start_pos = start_pos if start_pos is not None else 0

        if isinstance(start_pos, int):
            position_ids = position_ids + start_pos
        else:
            position_ids = position_ids + start_pos[:, None]

        hidden_states = self.embed_tokens(input_ids).bfloat16()
        
        pos_emb = self.rotary_emb(position_ids, hidden_states.dtype)

        for i in range(self.config.num_hidden_layers):
            hidden_states = self.layers[i].infer(
                hidden_states,
                pos_emb,
                start_pos,
                item_indexes,
            )

        hidden_states = self.norm(hidden_states)

        if self.lm_head:
            weight = self.lm_head.weight
        else:
            weight = self.embed_tokens.weight

        return nn.functional.linear(hidden_states.bfloat16(), weight.bfloat16()).to(hidden_states.dtype)

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(
                max_batch_size, max_seq_len, device=device
            )

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @torch.no_grad()
    def generate(
        self, 
        input_ids: list[list[int]], 
        max_gen_len: int,
        temperature: float
    ) -> list[list[int]]:
        config = self.config
        bs = len(input_ids)

        input_lens = torch.tensor([*map(len, input_ids)], dtype=torch.int64, device=self.device)
        print("input_lens=", input_lens)
        max_input_len = input_lens.max().item()

        self.init_kv_cache(bs, max_input_len + max_gen_len, self.device)
        
        self.eval()

        output_ids = [[] for _ in range(bs)]
        props_cache = {}
        ids_to_indexes = {}

        for k in range(bs):
            print(
                f"\r* Prefilling: {k+1:>4d}/{bs:>4d}",
                flush=True,
                end="",
            )

            key = '|'.join(map(str, input_ids[k]))

            if key in props_cache:
                probs = props_cache[key]
                k_prev = ids_to_indexes[key]
                for layer_idx in range(self.config.num_hidden_layers):
                    self.layers[layer_idx].self_attn.key_cache[k] = self.layers[layer_idx].self_attn.key_cache[k_prev] 
                    self.layers[layer_idx].self_attn.value_cache[k] = self.layers[layer_idx].self_attn.value_cache[k_prev] 
            else:
                logits = self.infer(
                    input_ids= torch.tensor([input_ids[k]], dtype=torch.int64, device=self.device),
                    start_pos=None,
                    item_indexes=torch.tensor([k], dtype=torch.int64, device=self.device)
                )
                probs = torch.softmax(logits[0, -1], dim=-1)
                props_cache[key] = probs
                ids_to_indexes[key] = k

            output_ids[k].append(torch.multinomial(probs, num_samples=1)[0].item())

        finish_indexes = set()
        last_token_ids = torch.tensor([t[-1] for t in output_ids], dtype=torch.int64, device=self.device)

        for i in range(max_gen_len - 1):
            print(
                f"\r* Generating trajectories: {i+1:>4d}/{max_gen_len:>4d}",
                flush=True,
                end="",
            )
                
            #with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            unique_ids = []
            unique_indexes = []
            
            for k in range(bs):
                if k in finish_indexes: 
                    continue
                key = '|'.join(map(str, input_ids[k] + output_ids[k]))

                if key not in unique_ids:
                    unique_ids.append(key)
                    unique_indexes.append(k)

            item_indexes = torch.tensor(unique_indexes, dtype=torch.int64, device=self.device)
            logits = self.infer(last_token_ids[item_indexes, None], input_lens[item_indexes] + i, item_indexes)
            logits = logits.float()
            logits = logits[:,-1]
                
            probs = torch.softmax(logits/temperature, dim=-1)
            last_token_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)

            for k in range(bs):
                if k in finish_indexes: continue

                key = '|'.join(map(str, input_ids[k] + output_ids[k]))
                idx = unique_ids.index(key)
                k_prev = unique_indexes[idx]

                if k != k_prev:
                    cache_pos = input_lens[k] + i
                    for layer_idx in range(self.config.num_hidden_layers):
                        self.layers[layer_idx].self_attn.key_cache[k,cache_pos] = self.layers[layer_idx].self_attn.key_cache[k_prev,cache_pos] 
                        self.layers[layer_idx].self_attn.value_cache[k,cache_pos] = self.layers[layer_idx].self_attn.value_cache[k_prev,cache_pos] 

                new_token_id = torch.multinomial(probs[idx], num_samples=1)[0].item()

                if new_token_id == config.eos_token_id:
                    finish_indexes.add(k)

                if k not in finish_indexes:
                    output_ids[k].append(new_token_id)
                    
                last_token_ids[k] = new_token_id

            if len(finish_indexes) == bs:
                break

        self.del_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return output_ids
    
    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        import json
        import safetensors.torch
        with open(ckpt_path + "/config.json", "r") as f:
            config = json.load(f)
        if not config.get("head_dim"):
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = QwenConfig(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            tie_word_embeddings=config["tie_word_embeddings"],
            awq_group_size=config.get("quantization_config", {}).get("group_size", 128),
            use_bf_scales=config.get("torch_dtype") == "bfloat16",
            version="2" if "Qwen2" in config["architectures"][0] else "3",
        )

        with torch.device("meta"):
            model = cls(config, device=device)

        files = glob.glob(ckpt_path + "/*.safetensors")
        weights = {}

        for f in files:
            weights.update(safetensors.torch.load_file(f, device="cpu"))

        states = {k.replace('model.',''):v for k,v in weights.items()}

        key_prefixes = [k.replace('.qweight', '') for k in states if k.endswith('.qweight')]
        awq_weights = {}

        for key_prefix in key_prefixes:
            qweight = states.pop(key_prefix + ".qweight")
            qzeros = states.pop(key_prefix + ".qzeros")
            scales = states.pop(key_prefix + ".scales")
            print("Convert weights:", key_prefix)
            awq_weights[key_prefix + ".qweight"] = cvt_qweights(qweight)
            awq_weights[key_prefix + ".qzeros"] = cvt_qzeros(qzeros)
            awq_weights[key_prefix + ".scales"] = scales.to(torch.bfloat16 if config.use_bf_scales else torch.float16)

        states = {key: value.to(torch.bfloat16) for key, value in states.items()}
        states.update(awq_weights)
        states = {k:v for k,v in states.items() if k in model.state_dict()}

        model.load_state_dict(states, strict=True, assign=True)
        return model.to(device)
