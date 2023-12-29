import torch
import torch.nn as nn
m_dim=300
dictsize=500
num_head=4
dim_ff=512
dropout=0.1
layernum=4
class Encoder(nn.Module):
    def __init__(self, lang):
        super(Encoder, self).__init__()
        self.emb=nn.Embedding(dictsize, m_dim)
        self.enc_layer=nn.TransformerEncoderLayer(d_model=m_dim, nhead=num_head, dim_feedforward=dim_ff, dropout=dropout)
        self.enc=nn.TransformerEncoder(self.enc_layer, layernum)
        self.dec_layer=nn.TransformerDecoderLayer(d_model=m_dim, nhead=num_head, dim_feedforward=dim_ff, dropout=dropout)
        self.dec=nn.TransformerDecoder(self.dec_layer, layernum)

class Decoder(nn.Module):
    def __init__(self, lang):
        self.cls=nn.Linear(m_dim, dictsize)
        self.enc_layer=nn.TransformerEncoderLayer(d_model=m_dim, nhead=num_head, dim_feedforward=dim_ff, dropout=dropout)
        self.enc=nn.TransformerEncoder(self.enc_layer, layernum)
        self.dec_layer=nn.TransformerDecoderLayer(d_model=m_dim, nhead=num_head, dim_feedforward=dim_ff, dropout=dropout)
        self.dec=nn.TransformerDecoder(self.dec_layer, layernum)
    

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.emb=nn.Embedding(dictsize, m_dim)
        self.cls=nn.Linear(m_dim, dictsize)

def comp(model, emb, tokseq, maxseq):
    enc=model
    seq=[]
    tokseq=emb.emb(tokseq)
    encout=enc.enc(tokseq)
    for _ in range(maxseq):
        seq.append(enc.dec(seq, encout)[0])
    seq=torch.stack(seq)
    #eos 이후의 값을 0으로 수정하여야 할 것.
    return seq

def reconstruct(model,embmod, embed):
    dec=model
    seq=[]
    encout=dec.enc(embed)

    eos=3

    while True:
        decout=dec.dec(seq, encout)[0]
        decout=embmod.cls(decout)
        seq.append(decout)
        if decout==eos:
            break
    seq=torch.stack(seq)
    return seq


def train_multi_enc(krtokenseq, entokenseq, jptokenseq):
    enc_kr=Encoder()
    enc_en=Encoder()
    enc_jp=Encoder()
    lossfn=nn.CrossEntropyLoss()
    optim=torch.optim.Adam(list(enc_kr.parameters()) + list(enc_en.parameters()) + list(enc_jp.parameters()), 0.0001)
    
    emb_kr=comp(enc_kr(), krtokenseq)
    emb_en=comp(enc_en(), entokenseq)
    emb_jp=comp(enc_jp(), jptokenseq)
    #결과물에 padding 추가해서 가장 긴 것을 기준으로 맞춘다.

    loss=lossfn(emb_kr, emb_en)+lossfn(emb_en, emb_jp)+lossfn(emb_jp, emb_kr)
    optim.zero_grad()
    loss.backward()
    optim.step()

def train_reconstruction(lang):
    enc=Encoder(lang=lang)
    dec=Decoder(lang=lang)
    
    emb=enc(tokenseq)
    outseq=dec(emb)

def train_backtranslate():