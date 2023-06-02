import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
import nmt_dataset

current_path = os.getcwd()
# envi_folder_path = r"dataset\en-vi"
envi_folder_path = os.path.join(current_path, "dataset/en-vi")
# envi_folder_path = "/home/hnc/PycharmProjects/vdt_project1/dataset/en-vi"
use_gpu = torch.cuda.is_available()
# envi_folder_path = "/home/hnc/PycharmProjects/nlp_demo/dataset/en-vi-dep"


def is_interative_notebook():
    return __name__ == "__main__"

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_head, tgt_head, src_dep, tgt_dep, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_head, src_dep, src_mask), src_mask, tgt, tgt_mask, tgt_head, tgt_dep)

    def encode(self, src, src_head, src_dep, src_mask):
        # print(src.shape, src_head.shape, src_dep.shape, src_mask.shape)
        return self.encoder(self.src_embed([src, src_head, src_dep]), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, tgt_head, tgt_dep):
        return self.decoder(self.tgt_embed([tgt, tgt_head, tgt_dep]), memory, src_mask, tgt_mask)

    # def forward(self, src, tgt, src_mask, tgt_mask):
    #     return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # def encode(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)

    # def decode(self, memory, src_mask, tgt, tgt_mask):
    #     return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, vocab_dep):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.lut_head = nn.Embedding(vocab, d_model)
        self.lut_dep = nn.Embedding(vocab_dep, d_model)
        self.d_model = d_model

    def forward(self, y):
        x, x_head, x_dep = y
        emb = self.lut(x)
        emb_head = self.lut_head(x_head)
        emb_dep = self.lut_dep(x_dep)
        emb = emb + emb_head + emb_dep
        return emb * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, src_vocab_dep, tgt_vocab_dep, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab, src_vocab_dep), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab, tgt_vocab_dep), c(position)),
        Generator(d_model, tgt_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if use_gpu:
        model.cuda(torch.device("cuda"))

    return model

class Batch:
    def __init__(self, src, tgt, src_head, tgt_head, src_dep, tgt_dep, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

        self.src_head = src_head
        self.src_mask_head = (src_head != pad).unsqueeze(-2)
        if tgt_head is not None:
            self.tgt_head = tgt_head[:, :-1]
            self.tgt_y_head = tgt_head[:, 1:]
            self.tgt_mask_head = self.make_std_mask(self.tgt_head, pad)
            self.ntokens_head = (self.tgt_y_head != pad).data.sum()

        self.src_dep = src_dep
        self.src_mask_dep = (src_dep != pad).unsqueeze(-2)
        if tgt_head is not None:
            self.tgt_dep = tgt_dep[:, :-1]
            self.tgt_y_dep = tgt_dep[:, 1:]
            self.tgt_mask_dep = self.make_std_mask(self.tgt_dep, pad)
            self.ntokens_dep = (self.tgt_y_dep != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0

def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_head, batch.tgt_head, batch.src_dep, batch.tgt_dep, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node

    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1

    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step*warmup**(-1.5)))

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x/d, 1/d, 1/d, 1/d]])
    return crit(predict.log(), torch.LongTensor([1])).data

def data_gen(V, batch_size, nbatches):
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.require_grad_(False).clone().detach()
        tgt = data.require_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))/norm)
        return sloss.data*norm, sloss

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def load_tokenizers():
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_en):
    def tokenize_vi(text):
        return text.split()

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    train = nmt_dataset.VLSP2021(envi_folder_path, "train")
    val = nmt_dataset.VLSP2021(envi_folder_path, "dev")
    test = nmt_dataset.VLSP2021(envi_folder_path, "tst")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )

    train = nmt_dataset.VLSP2021(envi_folder_path, "train")
    val = nmt_dataset.VLSP2021(envi_folder_path, "dev")
    test = nmt_dataset.VLSP2021(envi_folder_path, "tst")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_vi, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )

    train = nmt_dataset.VLSP2021_DEP(envi_folder_path, "train")
    val = nmt_dataset.VLSP2021_DEP(envi_folder_path, "dev")
    test = nmt_dataset.VLSP2021_DEP(envi_folder_path, "tst")
    vocab_src_dep = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_vi, index=0),
        min_freq=2
    )

    train = nmt_dataset.VLSP2021_DEP(envi_folder_path, "train")
    val = nmt_dataset.VLSP2021_DEP(envi_folder_path, "dev")
    test = nmt_dataset.VLSP2021_DEP(envi_folder_path, "tst")
    vocab_tgt_dep = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_vi, index=1),
        min_freq=2
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep

def load_vocab(spacy_en):
    if not exists("vocab_envi.pt"):
        vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep = build_vocabulary(spacy_en)
        torch.save((vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep), "vocab_envi.pt")
    else:
        vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep = torch.load("vocab_envi.pt")

    return vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep

if is_interative_notebook():
    spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep = show_example(load_vocab, args=[spacy_en])

def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        src_vocab_dep,
        tgt_vocab_dep,
        device,
        max_padding=128,
        pad_id=2,
):
    bs_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list, tgt_list, src_list_head, tgt_list_head, src_list_dep, tgt_list_dep = [], [], [], [], [], []
    for (_src, _tgt) in batch:
        if 'head : ' in _src:
            processed_src_head = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        src_vocab(src_pipeline(_src[7:])),
                        dtype=torch.int64,
                        device=device
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt_head = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        tgt_vocab(tgt_pipeline(_tgt[7:])),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list_head.append(
                pad(processed_src_head, (
                    0, max_padding - len(processed_src_head)
                ), value=pad_id, )
            )
            tgt_list_head.append(
                pad(processed_tgt_head, (
                    0, max_padding - len(processed_tgt_head)
                ), value=pad_id, )
            )
        elif 'dep : ' in _src:
            processed_src_dep = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        src_vocab_dep(_src[6:].split()),
                        dtype=torch.int64,
                        device=device
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt_dep = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        tgt_vocab_dep(_tgt[6:].split()),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list_dep.append(
                pad(processed_src_dep, (
                    0, max_padding - len(processed_src_dep)
                ), value=pad_id, )
            )
            tgt_list_dep.append(
                pad(processed_tgt_dep, (
                    0, max_padding - len(processed_tgt_dep)
                ), value=pad_id, )
            )
        else:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        src_vocab(src_pipeline(_src)),
                        dtype=torch.int64,
                        device=device
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        tgt_vocab(tgt_pipeline(_tgt)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                pad(processed_src,(
                    0, max_padding-len(processed_src)
                ), value=pad_id,)
            )
            tgt_list.append(
                pad(processed_tgt, (
                    0, max_padding - len(processed_tgt)
                ), value=pad_id,)
            )

    max_size = max(len(src_list), len(src_list_head), len(src_list_dep), len(tgt_list), len(tgt_list_head), len(tgt_list_dep))
    padding_tensor = torch.zeros(max_padding, dtype=torch.int64, device=device)

    while (len(src_list) < max_size):
        src_list.append(padding_tensor)
    while (len(tgt_list) < max_size):
        tgt_list.append(padding_tensor)
    while (len(src_list_head) < max_size):
        src_list_head.append(padding_tensor)
    while (len(tgt_list_head) < max_size):
        tgt_list_head.append(padding_tensor)
    while (len(src_list_dep) < max_size):
        src_list_dep.append(padding_tensor)
    while (len(tgt_list_dep) < max_size):
        tgt_list_dep.append(padding_tensor)



    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    src_head = torch.stack(src_list_head)
    tgt_head = torch.stack(tgt_list_head)

    src_dep = torch.stack(src_list_dep)
    tgt_dep = torch.stack(tgt_list_dep)
    # print(max_size)
    # print("Collect batch", src.shape, src_head.shape, src_dep.shape)
    return (src, tgt, src_head, tgt_head, src_dep, tgt_dep)

def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        vocab_src_dep,
        vocab_tgt_dep,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True
):
    def tokenize_vi(text):
        return text.split()

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_vi,
            vocab_src,
            vocab_tgt,
            vocab_src_dep,
            vocab_tgt_dep,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter = nmt_dataset.VLSP2021(envi_folder_path, "train")
    valid_iter = nmt_dataset.VLSP2021(envi_folder_path, "dev")
    test_iter = nmt_dataset.VLSP2021(envi_folder_path, "tst")

    train_iter_head = nmt_dataset.VLSP2021_HEAD(envi_folder_path, "train", prefix='head')
    valid_iter_head = nmt_dataset.VLSP2021_HEAD(envi_folder_path, "dev", prefix='head')
    test_iter_head = nmt_dataset.VLSP2021_HEAD(envi_folder_path, "tst", prefix='head')

    train_iter_dep = nmt_dataset.VLSP2021_DEP(envi_folder_path, "train", prefix='dep')
    valid_iter_dep = nmt_dataset.VLSP2021_DEP(envi_folder_path, "dev", prefix='dep')
    test_iter_dep = nmt_dataset.VLSP2021_DEP(envi_folder_path, "tst", prefix='dep')

    train_iter_map = to_map_style_dataset(train_iter + train_iter_head + train_iter_dep)
    train_sampler = (DistributedSampler(train_iter_map) if is_distributed else None)

    valid_iter_map = to_map_style_dataset(valid_iter + valid_iter_head + valid_iter_dep)
    valid_sampler = (DistributedSampler(valid_iter_map) if is_distributed else None)

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn
    )
    return train_dataloader, valid_dataloader

def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        vocab_src_dep,
        vocab_tgt_dep,
        spacy_en,
        config,
        is_distributed=False,
):
    if use_gpu:
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), len(vocab_src_dep), len(vocab_tgt_dep), N=6)
    if use_gpu:
        model.cuda(gpu)

    module = model
    is_main_process = True
    # if is_distributed:
    #     dist.init_process_group("nccl", init_method="env://", rank=gpu,
    #                             world_size=ngpus_per_node)
    #     model = DDP(model, device_ids=[gpu])
    #     module = model.module
    #     is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    if use_gpu:
        criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu if use_gpu else torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        vocab_src_dep,
        vocab_tgt_dep,
        spacy_en,
        batch_size=config["batch_size"]//ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()
    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)
        model.train()

        _, train_state = run_epoch(
            (Batch(b[0], b[1], b[2], b[3], b[4], b[5], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)

        torch.cuda.empty_cache()

        model.eval()

        sloss = run_epoch(
            (Batch(b[0], b[1], b[2], b[3], b[4], b[5], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

def train_model(vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep, spacy_en, config):
    # if config["distributed"]:
    #     train_distributed_model(vocab_src, vocab_tgt, spacy_en, config)
    # else:
    train_worker(0, 1, vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep, spacy_en, config, False)

def load_trained_model():
    config = {
        "batch_size": 8,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "vlsp_model_"
    }
    model_path = "vlsp_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, vocab_src_dep, vocab_tgt_dep, spacy_en, config)

    # model_path = "vlsp_model_00.pt"

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    if use_gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model

if is_interative_notebook():
    model = load_trained_model()

# if False:
#     model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
#     model.generator.lut.weight = model.tgt_embed[0].lut.weight

def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

def check_outputs(
        valid_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        n_examples=15,
        pad_idx=2,
        eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]
        with open("reference.txt", 'a') as f:
            f.write(
            " ".join(tgt_tokens).replace("\n", "") + "\n")
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]).split(eos_string, 1)[0] + eos_string
        )
        with open("hypothesis.txt", 'a') as f:
            f.write(model_txt + "\n")

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

    return results

def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_en
    _, valid_dataloader = create_dataloaders(
        torch.device("cuda") if use_gpu else torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_en,
        batch_size=1,
        is_distributed=False
    )

    model_path = "vlsp_model_final.pt"
    # model_path = "vlsp_model_00.pt"

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    if use_gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    example_data = check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)

    return model, example_data

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return(
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn

def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn

def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn

def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))

def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )

show_example(viz_encoder_self)

def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

# show_example(viz_decoder_self)