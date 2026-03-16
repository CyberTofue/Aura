"""
Sensor AI Server
Serves two models:
  POST /image/generate  — PicoGen image generation
  POST /chat/generate   — Aura text generation
  GET  /health          — health check
"""

import os, io, math, base64, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sentencepiece as spm

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_CKPT = os.environ.get('IMAGE_CKPT', 'image_checkpoint.pth')
CHAT_CKPT  = os.environ.get('CHAT_CKPT',  'chat_checkpoint.pth')
TOKENIZER  = os.environ.get('TOKENIZER',  'tokenizer.model')

# ═══════════════════════════════════════════════════════════════
# IMAGE MODEL  (PicoGen)
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        hidden = (int(2 * mult * dim / 3) + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6*dim, bias=True))
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)
    def forward(self, x, cond):
        return (self.norm(x),) + self.proj(cond).unsqueeze(1).chunk(6, dim=-1)

class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult):
        super().__init__()
        self.adaLN = AdaLN(dim, dim)
        self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim); self.k_norm = RMSNorm(self.head_dim)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ffn = SwiGLU(dim, ffn_mult)
    def forward(self, x, cond):
        B, N, D = x.shape; H = self.n_heads
        x_n, s1, sh1, g1, s2, sh2, g2 = self.adaLN(x, cond)
        q, k, v = self.qkv(x_n*(1+s1)+sh1).chunk(3, dim=-1)
        q = q.view(B,N,H,self.head_dim).transpose(1,2)
        k = k.view(B,N,H,self.head_dim).transpose(1,2)
        v = v.view(B,N,H,self.head_dim).transpose(1,2)
        attn = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        x = x + g1 * self.attn_out(attn.transpose(1,2).contiguous().view(B,N,D))
        return x + g2 * self.ffn(nn.functional.layer_norm(x, (D,), eps=1e-6)*(1+s2)+sh2)

class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:,None] * freq[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))

class LabelEncoder(nn.Module):
    def __init__(self, vocab_size, dim, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads,
            dim_feedforward=dim*4, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = RMSNorm(dim); self.pool = nn.Linear(dim, dim)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, ids, drop_mask=None):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device)).unsqueeze(0)
        cond = self.pool(self.norm(self.enc(x)).mean(dim=1))
        if drop_mask is not None:
            cond = torch.where(drop_mask.unsqueeze(1), self.null.expand(B,1,-1).squeeze(1), cond)
        return cond

class PicoGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg['dim']; ps = cfg['patch_size']
        C = cfg['img_channels']; S = cfg['img_size']
        patch_dim = ps * ps * C
        self.patch_emb = nn.Linear(patch_dim, dim)
        pos = self._sincos2d(S // ps, dim)
        self.register_buffer('pos_enc', pos)
        self.time_emb  = TimeEmbed(dim)
        self.label_enc = LabelEncoder(cfg['vocab_size'], dim, max_len=cfg['max_label_len'])
        self.cond_proj = nn.Sequential(nn.Linear(dim*2, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, cfg['n_heads'], cfg['ffn_mult']) for _ in range(cfg['n_layers'])])
        self.final_norm  = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2*dim))
        self.final_proj  = nn.Linear(dim, patch_dim)

    @staticmethod
    def _sincos2d(n, dim):
        y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        y, x = y.flatten().float(), x.flatten().float()
        d = dim // 4
        freq = 1.0 / (10000 ** (torch.arange(d).float() / d))
        return torch.cat([torch.sin(y[:,None]*freq), torch.cos(y[:,None]*freq),
                          torch.sin(x[:,None]*freq), torch.cos(x[:,None]*freq)], dim=-1)

    def _patchify(self, imgs):
        B, C, H, W = imgs.shape; ps = self.cfg['patch_size']
        x = imgs.unfold(2,ps,ps).unfold(3,ps,ps)
        x = x.contiguous().view(B,C,-1,ps,ps).permute(0,2,1,3,4).contiguous()
        return x.view(B, x.size(1), -1)

    def _unpatchify(self, x):
        B = x.shape[0]; ps = self.cfg['patch_size']
        C = self.cfg['img_channels']; nh = nw = self.cfg['img_size'] // ps
        return x.view(B,nh,nw,C,ps,ps).permute(0,3,1,4,2,5).contiguous().view(B,C,nh*ps,nw*ps)

    def forward(self, x, t, label_ids, drop_mask=None):
        h = self.patch_emb(self._patchify(x)) + self.pos_enc
        cond = self.cond_proj(torch.cat([self.time_emb(t), self.label_enc(label_ids, drop_mask)], dim=-1))
        for blk in self.blocks: h = blk(h, cond)
        sc, sh = self.final_adaLN(cond).unsqueeze(1).chunk(2, dim=-1)
        return self._unpatchify(self.final_proj(self.final_norm(h)*(1+sc)+sh))

    @torch.no_grad()
    def sample(self, label_ids, n_steps=50, cfg_scale=6.0):
        self.eval()
        B = label_ids.shape[0]; device = label_ids.device
        C, H, W = self.cfg['img_channels'], self.cfg['img_size'], self.cfg['img_size']
        x = torch.randn(B, C, H, W, device=device)
        null = torch.zeros_like(label_ids)
        dt = 1.0 / n_steps
        for t_val in torch.linspace(0.0, 1.0-dt, n_steps, device=device):
            t_b = torch.full((B,), t_val, device=device)
            v = self.forward(x, t_b, null) + cfg_scale * (
                self.forward(x, t_b, label_ids) - self.forward(x, t_b, null))
            x = x + v * dt
        return x.clamp(-1, 1)

class ImageVocab:
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz !?,.-\'()_'
    def __init__(self):
        self.idx2ch = ['<PAD>', '<BOS>', '<EOS>'] + list(self.CHARS)
        self.ch2idx = {c: i for i, c in enumerate(self.idx2ch)}
    def __len__(self): return len(self.idx2ch)
    def encode(self, t): return [self.ch2idx.get(c, 0) for c in t.lower()]

# ═══════════════════════════════════════════════════════════════
# CHAT MODEL  (Aura)
# ═══════════════════════════════════════════════════════════════

N_EMBD           = 256
N_LAYER          = 6
N_HEAD           = 4
N_KV_HEAD        = 2
N_EXPERTS        = 8
N_EXPERTS_ACTIVE = 2
BLOCK_SIZE       = 384

class GQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head    = N_HEAD
        self.n_kv_head = N_KV_HEAD
        self.n_rep     = N_HEAD // N_KV_HEAD
        self.head_dim  = N_EMBD // N_HEAD
        self.q_proj    = nn.Linear(N_EMBD, N_HEAD    * self.head_dim, bias=False)
        self.k_proj    = nn.Linear(N_EMBD, N_KV_HEAD * self.head_dim, bias=False)
        self.v_proj    = nn.Linear(N_EMBD, N_KV_HEAD * self.head_dim, bias=False)
        self.o_proj    = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.resid_drop = nn.Dropout(0.0)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(BLOCK_SIZE).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', torch.cos(emb))
        self.register_buffer('sin_cached', torch.sin(emb))

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def apply_rope(self, x, T):
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self.apply_rope(q, T)
        k = self.apply_rope(k, T)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.resid_drop(self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        hidden  = int(8/3 * N_EMBD)
        self.w1 = nn.Linear(N_EMBD, hidden, bias=False)
        self.w2 = nn.Linear(hidden, N_EMBD, bias=False)
        self.w3 = nn.Linear(N_EMBD, hidden, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(N_EXPERTS)])
        self.router  = nn.Linear(N_EMBD, N_EXPERTS, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        flat = x.view(-1, C)
        probs = F.softmax(self.router(flat), dim=-1)
        topk_probs, topk_idx = torch.topk(probs, N_EXPERTS_ACTIVE, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        output = torch.zeros_like(flat)
        for k in range(N_EXPERTS_ACTIVE):
            for e_idx in range(N_EXPERTS):
                mask = (topk_idx[:, k] == e_idx)
                if not mask.any(): continue
                output[mask] += topk_probs[mask, k].unsqueeze(-1) * self.experts[e_idx](flat[mask])
        return output.view(B, T, C)

class ChatBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(N_EMBD)
        self.attn = GQAttention()
        self.ln_2 = RMSNorm(N_EMBD)
        self.mlp  = MoELayer()
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class AuraModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, N_EMBD),
            drop = nn.Dropout(0.0),
            h    = nn.ModuleList([ChatBlock() for _ in range(N_LAYER)]),
            ln_f = RMSNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
    def forward(self, idx):
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h: x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

# ═══════════════════════════════════════════════════════════════
# STARTUP — load both models
# ═══════════════════════════════════════════════════════════════

image_model = None
image_vocab = None
chat_model  = None
sp          = None
BOS_ID = SEP_ID = EOS_ID = None

@app.on_event('startup')
def load_models():
    global image_model, image_vocab, chat_model, sp, BOS_ID, SEP_ID, EOS_ID

    # Image model
    if os.path.exists(IMAGE_CKPT):
        print(f'Loading image model from {IMAGE_CKPT}...')
        ckpt = torch.load(IMAGE_CKPT, map_location=DEVICE, weights_only=False)
        cfg  = ckpt['cfg']
        image_vocab = ImageVocab()
        cfg['vocab_size'] = len(image_vocab)
        image_model = PicoGen(cfg).to(DEVICE)
        image_model.load_state_dict(ckpt['model'])
        image_model.eval()
        print(f'Image model loaded — epoch {ckpt["epoch"]}  loss {ckpt["loss"]:.4f}')
    else:
        print(f'WARNING: image checkpoint not found at {IMAGE_CKPT}')

    # Chat model
    if os.path.exists(CHAT_CKPT) and os.path.exists(TOKENIZER):
        print(f'Loading chat model from {CHAT_CKPT}...')
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER)
        BOS_ID = sp.piece_to_id('<|bos|>')
        SEP_ID = sp.piece_to_id('<|sep|>')
        EOS_ID = sp.piece_to_id('<|eos|>')
        ckpt   = torch.load(CHAT_CKPT, map_location=DEVICE, weights_only=False)
        state  = ckpt.get('model', ckpt)
        chat_model = AuraModel(sp.get_piece_size()).to(DEVICE)
        chat_model.load_state_dict(state, strict=False)
        chat_model.eval()
        print(f'Chat model loaded — step {ckpt.get("step","?")}')
    else:
        print(f'WARNING: chat checkpoint or tokenizer not found')

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

class ImageRequest(BaseModel):
    prompt:    str
    n_steps:   int   = 50
    cfg_scale: float = 6.0

class ChatRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = 200
    temperature: float = 0.5
    top_k:       int   = 40
    rep_penalty: float = 1.2

@app.get('/health')
def health():
    return {
        'status':       'ok',
        'image_model':  image_model is not None,
        'chat_model':   chat_model  is not None,
        'device':       DEVICE,
    }

@app.post('/image/generate')
def image_generate(req: ImageRequest):
    if image_model is None:
        raise HTTPException(503, 'Image model not loaded')

    vocab = image_vocab
    ml    = image_model.cfg['max_label_len']
    ids   = [vocab.ch2idx['<BOS>']] + [vocab.ch2idx.get(c, 0) for c in req.prompt.lower()[:ml-2]] + [vocab.ch2idx['<EOS>']]
    ids  += [0] * (ml - len(ids))
    ids   = torch.tensor([ids[:ml]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        img = image_model.sample(ids, n_steps=req.n_steps, cfg_scale=req.cfg_scale)

    arr = ((img[0].cpu().float().clamp(-1,1) + 1) / 2 * 255).byte().numpy()
    pil = Image.fromarray(arr.transpose(1,2,0).astype('uint8'), 'RGB')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {'image_b64': b64, 'format': 'png'}

@app.post('/chat/generate')
@torch.no_grad()
def chat_generate(req: ChatRequest):
    if chat_model is None:
        raise HTTPException(503, 'Chat model not loaded')

    ids = [BOS_ID] + sp.encode(req.prompt) + [SEP_ID]
    idx = torch.tensor([ids], device=DEVICE)
    generated = []

    for _ in range(req.max_tokens):
        logits = chat_model(idx[:, -BLOCK_SIZE:])[:, -1, :]
        logits = logits / max(req.temperature, 1e-6)
        for tid in set(generated[-80:]):
            logits[0, tid] /= req.rep_penalty
        probs = F.softmax(logits, dim=-1)
        if req.top_k > 0:
            v, _ = torch.topk(probs, min(req.top_k, probs.size(-1)))
            probs[probs < v[:, [-1]]] = 0
            probs /= probs.sum(-1, keepdim=True)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == EOS_ID:
            break
        generated.append(nxt)
        idx = torch.cat((idx, torch.tensor([[nxt]], device=DEVICE)), dim=1)

    text = sp.decode(generated) if generated else ''
    return {'text': text, 'tokens': len(generated)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)