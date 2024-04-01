# Originally a notebook, but converted it to a python file

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Colab-Data/en-fr.csv')
df = df[df.apply(lambda x: len(str(x['en'])) < 50 and len(str(x['fr'])) < 50, axis=1)]
df.to_csv("filtered_file.csv", index=False)

import pandas as pd
from torch.utils.data import Dataset, DataLoader
TRAIN_DATA = 20000000
class get_datasets(Dataset):
    def __init__(self, split, language_pair = ('fr', 'en')):
        self.df = None
        if split == 'train':
            self.df = pd.read_csv('/content/drive/MyDrive/Colab-Data/en-fr.csv', nrows=TRAIN_DATA)
            print(len(self.df))
        else:
            self.df = pd.read_csv('/content/drive/MyDrive/Colab-Data/en-fr.csv', skiprows=TRAIN_DATA)
        if split == 'train':
           #FILTER OUT SOME DATA THAT WOULD BREAK GPU MEMORY, STILL ALMOST ALL THE DATA THOUGH, ONLY MINOR PARTS
           #FILTERED OUT 
          self.df = self.df[self.df.apply(lambda x: len(str(x['en'])) < 300 and len(str(x['fr'])) < 300, axis=1)]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        french = self.df.iloc[index, 1]
        english = self.df.iloc[index, 0]

        return str(english), str(french)

import torch
import gc
torch.cuda.empty_cache()
gc.collect()


import torch
import math
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.metrics import bleu_score
from matplotlib import pyplot as plt

import tqdm

source = 'en'
target = 'fr'
token_transform = {}
vocab_transform = {}
#TOKENIZER FOR VOCABULARY THAT IS GENERATED BASED ON DATA, USING SPACY
#TO TOKENIZE BUT WE ALSO TRIED WITH OUR OWN TOKENIZER OF ONE HOTS BUT
# WE FOUND SPACY TO CONTAIN BETTER RESULTS SO WE JUST PUT IT AS A BENCHMARK 
# AND FOR COMPARISON PURPOSES HERE 
token_transform[source] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[target] = get_tokenizer('spacy', language='fr_core_news_sm')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

Dataset = get_datasets
Training_Dataset = Dataset(split='train', language_pair=(source, target))
Validation_Dataset = Dataset(split='valid', language_pair=(source, target))

print(len(Training_Dataset))
print(len(Validation_Dataset))

#GETS TOKENS FROM THE VOCABULARY AND DATASET
def get_tokens(collection, language):
    lang_dict = {source: 0, target: 1}

    for data in collection:
        yield token_transform[language](data[lang_dict[language]])

unk, pad, bos, eos = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


for ln in [source, target]:
    train_iter = Training_Dataset
    vocab_transform[ln] = build_vocab_from_iterator(get_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [source, target]:
  vocab_transform[ln].set_default_index(unk)


en_vocab = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#POSITINAL ENCODING TO GIVE MODEL A SENSE OF SPACE
#THIS IS A COMMON TOKEN EMBEDDING ARCHITECTURE AMONG TRANSFORMER IMPLEMENTATIONS
class Encoding(torch.nn.Module):
    def __init__(self,
                 emb_size,
                 dropout,
                 maxlen=5000):
        super(Encoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size)) 
        pos_embedding[:, 0::2] = torch.sin(pos * den)  
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout =torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

#EMBEDDING LAYER
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

#TRANSFORMER LAYER
class Transformer(torch.nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward = 512,
                 dropout = 0.1):
        super(Transformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = torch.nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = Encoding(
            emb_size, dropout=dropout)
    #WE PASS THE SRC THROUGH THE ENCODING, THEN RUN A TRANSFORMER AND FINALLY A LINEAR LAYER 
    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    #ENCODES THE SRC STRING WITH A GIVEN MASK AND ACCOUNTS FOR POSITION 
    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)
    #DECODER 
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

#GENERATE A MASK FOR THE TOKENS
def generate_ss_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_ss_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad).transpose(0, 1)
    tgt_padding_mask = (tgt == pad).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

src_size = len(vocab_transform[source])
tgt_size = len(vocab_transform[target])
emb_size = 512
nhead = 8
ffn_hid_dim = 512
batch = 512
num_enc_layers = 3
num_dec_layers = 3

transformer = Transformer(num_enc_layers, num_dec_layers, emb_size,
                                 nhead, src_size, tgt_size, ffn_hid_dim)

for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

from torch.nn.utils.rnn import pad_sequence

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([bos]),
                      torch.tensor(token_ids),
                      torch.tensor([eos])))

text_transform = {}
for ln in [source, target]:
    text_transform[ln] = sequential_transforms(token_transform[ln], 
                                               vocab_transform[ln], 
                                               tensor_transform)

#FUNCTION TO TRANSFORM SOURCE DATA INTO BATCH TENSOR
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[source](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[target](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=pad)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Training_Dataset
    train_dataloader = DataLoader(train_iter, batch_size=batch, collate_fn=collate_fn)
    for (src, tgt) in train_dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Validation_Dataset
    val_dataloader = DataLoader(val_iter, batch_size=batch, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


#A COMMON DECODING BASED ON GREEDY SEARCH TO TRANSLATE DECODED
#TENSOR EMBEDDINGS INTO ACTUAL STRINGS 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_ss_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[source](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=bos).flatten()
    return " ".join(vocab_transform[target].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


#COMPUTES THE BLEU SCORE BASED ON THE DATA 
def compute_bleu_score(model: torch.nn.Module):
    candidate_corpus = []
    reference_corpus = []
    size = 0
    for src, tgt in Training_Dataset:
        size += 1
        translated = translate(model, src)
        candidate_corpus.append(translated.split())
        reference_corpus.append([tgt.split()])
        if size > 1000:
          break

    print('Bleu Score: ', bleu_score(candidate_corpus=candidate_corpus, references_corpus=reference_corpus))


from timeit import default_timer as timer
NUM_EPOCHS = 200
train_losses = []
epochs = [i for i in range(NUM_EPOCHS)]
for epoch in range(1, NUM_EPOCHS+1):
    print('CURRENT EPOCH', epoch)
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    score = compute_bleu_score(transformer)
    train_losses.append(score)

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}"f"Epoch time = {(end_time - start_time):.3f}s"))

plt.plot(epochs, train_losses, color='red', label='loss')
plt.xlabel('epochs')
plt.ylabel('bleu')
plt.title('bleu score')
plt.legend()
plt.show()

compute_bleu_score(transformer)
print(translate(transformer, "I love baguettes."))
