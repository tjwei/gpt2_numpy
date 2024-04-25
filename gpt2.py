# %%
import numpy as np
import tiktoken
import json
import time
from collections import namedtuple

enc = tiktoken.get_encoding("gpt2")
# curl -JL https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors -o gpt2.safetensors
# curl -JL https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors -o gpt2-medium.safetensors
# curl -JL https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors -o gpt2-large.safetensors
# curl -JL https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors -o gpt2-xl.safetensors

to_numpy_dtype = {
    "F32": np.float32,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
}

Weights = namedtuple("WEIGHT", ["W", "b"])


def get_array(byte_buffer, shape: list[int], dtype: str, data_offsets: list[int, int]):
    numpy_dtype = to_numpy_dtype[dtype]
    begin, end = data_offsets
    return np.frombuffer(byte_buffer[begin:end], dtype=numpy_dtype).reshape(shape)


def read_weights(model_name: str) -> dict:
    f = open(f"{model_name}.safetensors", "rb")
    N = int.from_bytes(f.read(8), byteorder="little")
    header = json.loads(f.read(N).decode("utf-8"))
    byte_buffer = f.read()
    weights = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        parent, node = None, weights
        name_split = k.split(".")
        for child_name in name_split[:-1]:
            if child_name not in node:
                node[child_name] = {}
            parent, node = node, node[child_name]
        leaf_name = name_split[-1]
        node[leaf_name] = get_array(byte_buffer, **v)
        if node.keys() == {"weight", "bias"} and parent is not None:
            parent[name_split[-2]] = Weights(node["weight"], node["bias"])
    return weights


class LayerNorm:
    def __init__(self, weight: Weights, eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        W, b = self.weight
        eps = self.eps
        xn = (x - x.mean(axis=2, keepdims=True)) / np.sqrt(
            x.var(axis=2, keepdims=True) + eps
        )
        return xn * W + b


class Linear:
    def __init__(self, weight: Weights):
        self.weight = weight

    def __call__(self, x):
        W, b = self.weight
        return x @ W + b


class Embedding:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return self.weight[x]


def GELU(x):
    a = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0)))
    return 0.5 * x * (1.0 + a)


def softmax(a):
    a = a - np.max(a, axis=-1, keepdims=True)
    a = np.exp(a)
    return a / a.sum(axis=-1, keepdims=True)


def to_heads(x, n_heads):
    return x.reshape(*x.shape[:2], n_heads, -1)


class SelfAttention:
    def __init__(self, weights: dict[str, Weights], n_heads, mask=None):
        self.c_attn_weight = weights["c_attn"]
        self.c_proj_weight = weights["c_proj"]
        n_embd = self.c_attn_weight.W.shape[1] // 3
        self.num_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.mask = mask
        self.k_cache = None
        self.v_cache = None

    def __call__(self, x, increamental=0):
        W1, b1 = self.c_attn_weight
        W2, b2 = self.c_proj_weight
        # x: [batch_size, seq_len, n_embd]
        y = (x @ W1) + b1
        # y: [batch_size, seq_len, n_embd*3]
        batch_size, seq_len = y.shape[:2]
        q, k, v = np.split(y, 3, axis=2)
        # q: [batch_size, seq_len, n_embd]
        k, q, v = map(lambda x: to_heads(x, self.num_heads), (k, q, v))
        # q: [batch_size, seq_len, num_heads=12, head_dim=64]
        if increamental:
            seq_len = increamental + 1
            k = np.concatenate([self.k_cache, k], axis=1)
            v = np.concatenate([self.v_cache, v], axis=1)
        self.k_cache = k
        self.v_cache = v
        attention_map = np.einsum("bmhd,bnhd->bmhn", q, k) / np.sqrt(self.head_dim)
        if (not increamental) and self.mask is not None:
            attention_map = np.where(
                self.mask[:, :seq_len, :, :seq_len], attention_map, -np.inf
            )
        attention_map = softmax(attention_map)
        y = np.einsum("bmhn, bnhd -> bmhd", attention_map, v)
        # y: [batch_size, num_heads, seq_len, head_dim]
        y = y.reshape(batch_size, y.shape[1], -1)
        # y: [batch_size, seq_len, n_embd]
        y = (y @ W2) + b2
        return y


class Block:
    def __init__(self, weights: dict, n_heads, mask=None, eps=1e-5):
        self.ln_1 = LayerNorm(weights["ln_1"], eps)
        self.ln_2 = LayerNorm(weights["ln_2"], eps)
        self.attn = SelfAttention(weights["attn"], n_heads=n_heads, mask=mask)
        self.mlp_c_fc = Linear(weights["mlp"]["c_fc"])
        self.mlp_c_proj = Linear(weights["mlp"]["c_proj"])
        self.mlpf = lambda x: self.mlp_c_proj(GELU(self.mlp_c_fc(x)))

    def __call__(self, x, increamental=0):
        x = x + self.attn(self.ln_1(x), increamental=increamental)
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT:
    def __init__(self, weights, n_heads):
        self.wte = Embedding(weights["wte"]["weight"])
        self.wpe = Embedding(weights["wpe"]["weight"])
        context_len = self.wpe.weight.shape[0]
        mask = np.tril(np.ones((context_len, context_len), dtype=bool))[None, :, None]
        H = weights["h"]
        self.h = [Block(H[str(i)], n_heads=n_heads, mask=mask) for i in range(len(H))]
        self.ln_f = LayerNorm(weights["ln_f"])
        self.lm_head = weights["wte"]["weight"].T
        self.last_seq = None
        self.this_seq = None

    def __call__(self, x: list[int]):
        self.this_seq = x.copy()
        seq_n = len(x)
        if self.last_seq is not None and self.last_seq == x[:-1]:
            increamental = seq_n - 1
            pos_enc = self.wpe.weight[seq_n - 1]
            x = (self.wte.weight[x[-1]] + pos_enc)[None, :]
        else:
            increamental = 0
            pos_enc = self.wpe.weight[:seq_n]
            x = self.wte(x) + pos_enc
        x = x[None, ...]
        for block in self.h:
            x = block(x, increamental=increamental)
        x = self.ln_f(x)
        x = x @ self.lm_head
        self.last_seq = self.this_seq
        return x[0, -1]


# %%
model_name, n_layer, n_heads, n_embd = [
    ("gpt2", 12, 12, 768),
    ("gpt2-medium", 24, 16, 1024),
    ("gpt2-large", 36, 20, 1280),
    ("gpt2-xl", 48, 25, 1600),
][0]
print(model_name, n_layer, n_heads, n_embd)
weights = read_weights(model_name)
gpt = GPT(weights, n_heads)
# %%
prompt = """User: Which mountain is the highest in the world?
Assistant: The highest mountain is"""
x = enc.encode(prompt, allowed_special="all")
t0 = time.time()
temperture = 0.8
N = 100
top_k = 50
print(prompt, end="")
for i in range(N):
    logits = gpt(x) / temperture
    probs = softmax(logits)
    thr = sorted(probs)[-top_k]
    probs[probs < thr] = 0
    probs = probs / probs.sum()
    i = np.random.choice(np.arange(len(probs)), p=probs)
    token = enc.decode([i])
    x.append(i)
    print(token, end="")
print()
print(len(x) / (time.time() - t0))

# %%
