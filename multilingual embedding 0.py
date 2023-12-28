import torch
import torch.nn as nn

class Encoder(nn.Module):#그냥 트랜스포머. 인코더엔 토큰시퀀스, 디코더는 압축된 벡터 반환.
    def __init__(self, lang):
        super(Encoder, self).__init__()
        self.emb=nn.Embedding()
        self.l0=nn.Linear()
        self.l1=nn.Linear()
        self.l2=nn.Linear()
    def __forward__(self, word):
        emb=self.emb(idx)
        emb=self.l0(emb)
        emb=self.l1(emb)
        emb=self.l2(emb)
        return emb

class Decoder(nn.Module):
    def __init__(self, lang):
        self.l0=nn.Linear()
        self.l1=nn.Linear()
        self.l2=nn.Linear()
        self.er=nn.Linear()
    def __forward__(self, emb):
        emb=self.l0(emb)
        emb=self.l1(emb)
        emb=self.l2(emb)
        idx=self.argmax(self.softmax(self.er(emb)))
        return idx
    
def comp(model, tokseq, maxseq):
    enc=model
    seq=[]
    encout=enc.enc(tokseq)
    for _ in range(maxseq):
        seq.append(enc.dec(seq, encout)[0])
    seq=torch.stack(seq)
    #eos 이후의 값을 0으로 수정하여야 할 것.
    return seq

def train_multi_enc():
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

def train_dec2enc():