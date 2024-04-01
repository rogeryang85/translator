# Originally a notebook, but converted it to a python file

import os
import pandas as pd
import numpy as np
import Levenshtein

import torch
import torch.nn as nn
import torchnlp.nn.lock_dropout
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import gc
from torchsummaryX import summary
import wandb
from glob import glob

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.metrics import bleu_score
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.metrics import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt
import spacy

from tqdm.notebook import tqdm
import gc
from torchsummaryX import summary
from glob import glob

from typing import Union, Iterable
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import string
import unidecode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

from google.colab import drive
drive.mount('/content/drive')

config = {
    'batch_size': 96,
    'epochs': 10,
    'lr': 1e-3
}

class EnFr_Dataset(Dataset):
    def __init__(self, split, language_pair = ('fr', 'en')):
        self.df = None
        if split == 'train':
            self.df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/aa/en-fr.csv', nrows=300000).dropna()
        else:
            self.df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/aa/en-fr.csv', skiprows=3000000, nrows=30000).dropna()
        self.df = self.df[self.df.apply(lambda x: len(str(x[0])) < 60 and len(str(x[1])) < 30, axis=1)]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        french = self.df.iloc[index, 1]
        english = self.df.iloc[index, 0]
        return str(english), str(french)

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'
token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='fr_core_news_sm')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(DEVICE)
Dataset = EnFr_Dataset
Training_Dataset = Dataset(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
Validation_Dataset = Dataset(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

print(len(Training_Dataset))
print(len(Validation_Dataset))

def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

UNK_TOKEN = 0
SOS_TOKEN = 2
EOS_TOKEN = 3
PAD_TOKEN = 1

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = Training_Dataset
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_TOKEN)

VOCAB = vocab_transform[TGT_LANGUAGE]

from torch.nn.utils.rnn import pad_sequence

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_TOKEN]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_TOKEN])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], 
                                               vocab_transform[ln], 
                                               tensor_transform) 

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_len = list(map(lambda x : x.shape[0], src_batch))
    tgt_len = list(map(lambda x : x.shape[0], tgt_batch))
    src_batch = pad_sequence(src_batch, padding_value=PAD_TOKEN)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_TOKEN)
    src_batch = src_batch.transpose(0, 1)
    return torch.nn.functional.one_hot(src_batch, num_classes=len(vocab_transform[SRC_LANGUAGE])).float(), tgt_batch.transpose(0, 1), torch.tensor(src_len), torch.tensor(tgt_len)

train_dataloader = DataLoader(Training_Dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
test_dataloader = DataLoader(Validation_Dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

for example_batch in train_dataloader:
  a, b, c, d = example_batch
  print(a.shape)
  print(b.shape)
  print(c)
  break

class Listener(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Listener, self).__init__()

        self.base_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=encoder_hidden_size, bidirectional=True)

        self.LSTMs = torch.nn.Sequential(
            torch.nn.LSTM(encoder_hidden_size*2, encoder_hidden_size, num_layers=3, bidirectional=True),
        )

    def forward(self, x, x_lens):
        packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.base_lstm(packed)
        encoder_packed, _ = self.LSTMs(outputs)
        encoder_outputs, encoder_lens = pad_packed_sequence(encoder_packed, batch_first=True)

        return encoder_outputs, encoder_lens
    
def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

class Attention(torch.nn.Module):

    def __init__(self, encoder_hidden_size, decoder_output_size, projection_size):
        super(Attention, self).__init__()

        self.key_projection     = torch.nn.Linear(encoder_hidden_size *2, projection_size)
        self.value_projection   = torch.nn.Linear(encoder_hidden_size *2, projection_size)
        self.query_projection   = torch.nn.Linear(decoder_output_size, projection_size)

        self.softmax            = torch.nn.Softmax(dim=1)
    def set_key_value_mask(self, encoder_outputs, encoder_lens):

        _, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key      = self.key_projection(encoder_outputs)
        self.value    = self.value_projection(encoder_outputs)

        self.padding_mask     =  (torch.arange(encoder_max_seq_len).unsqueeze(0) >= encoder_lens.unsqueeze(1)).to(DEVICE)

    def forward(self, decoder_output_embedding):

        self.query         = self.query_projection(decoder_output_embedding)

        raw_weights        = torch.bmm(self.key, self.query.unsqueeze(2)).squeeze(2)
        masked_raw_weights = raw_weights.masked_fill_(self.padding_mask, -float('inf'))

        attention_weights  = self.softmax(masked_raw_weights/np.sqrt(self.key.shape[2]))
        context            = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)

        return context, self.query, attention_weights
    
class Speller(torch.nn.Module):

    def __init__(self, embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module= None):
        super().__init__()

        self.context_size       = attention_module.value_projection.weight.shape[0]
        self.decoder_output_size = decoder_output_size

        self.vocab_size         = vocab_size

        self.embedding          = nn.Embedding(vocab_size, embed_size, padding_idx = EOS_TOKEN)

        self.lstm_cells         = torch.nn.Sequential(torch.nn.LSTMCell(input_size = embed_size+self.context_size, hidden_size = decoder_hidden_size),
                                                      torch.nn.LSTMCell(input_size = decoder_hidden_size, hidden_size = decoder_output_size))

        self.char_prob          = torch.nn.Sequential(
                torch.nn.Linear(self.context_size+decoder_output_size,vocab_size))

        self.char_prob.weight   = self.embedding.weight 

        self.attention          = attention_module


    def forward(self, encoder_outputs, encoder_lens, y = None, tf_rate = 1):

        '''
        Args:
            embedding: Attention embeddings
            hidden_list: List of Hidden States for the LSTM Cells
        '''

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        timesteps = 500
        predictions = []

        char = torch.full((batch_size,), fill_value=SOS_TOKEN, dtype= torch.long).to(DEVICE)

        hidden_states   = [None]*len(self.lstm_cells)

        attention_plot          = []
        attention_weights       = torch.zeros(batch_size, encoder_max_seq_len) 

    
        if self.attention != None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)

        context, _, _            = self.attention(torch.zeros((batch_size, self.decoder_output_size)).to(DEVICE))

        for t in range(timesteps):

            char_embed = self.embedding(char)

            decoder_input_embedding = torch.cat((char_embed,context), axis=1)

            for i in range(len(self.lstm_cells)):
                hidden_states[i] = self.lstm_cells[i](decoder_input_embedding, hidden_states[i])
                decoder_input_embedding = hidden_states[i][0]

            decoder_output_embedding = hidden_states[-1][0]

            if self.attention != None:
                context, projected_query, attention_weights = self.attention(decoder_output_embedding) 
            attention_plot.append(attention_weights[0].detach().cpu())

            output_embedding     = torch.cat((projected_query, context), axis=1)
            char_prob            = self.char_prob(output_embedding)

            predictions.append(char_prob)

            char = torch.argmax(char_prob,1)

        attention_plot  = torch.stack(attention_plot, axis=0)
        predictions     = torch.stack(predictions, axis=1)

        return predictions, attention_plot
    
class LAS(torch.nn.Module):
    def __init__(self, input_size, encoder_hidden_size,
                 vocab_size, embed_size,
                 decoder_hidden_size, decoder_output_size,
                 projection_size= 128):

        super(LAS, self).__init__()

        self.encoder        = Listener(input_size, encoder_hidden_size)
        attention_module    = Attention(encoder_hidden_size, decoder_output_size, projection_size)
        self.decoder        = Speller(embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module)

    def forward(self, x, x_lens, y = None, tf_rate = 1):

        encoder_outputs, encoder_lens = self.encoder(x, x_lens) 
        predictions, attention_plot = self.decoder(encoder_outputs, encoder_lens, y, tf_rate)

        return predictions, attention_plot

model = LAS(len(vocab_transform[SRC_LANGUAGE]), 256, len(VOCAB), 256, 256, 128, 128)

model = model.to(DEVICE)
print(model)

summary(model,
        x= example_batch[0].to(DEVICE),
        x_lens= example_batch[2],
        y= example_batch[1].to(DEVICE))

a, b = model(x= example_batch[0].to(DEVICE),
        x_lens= example_batch[2],
        y= example_batch[1].to(DEVICE))
print(b.shape)

config['lr'] = 0.001

optimizer   = torch.optim.Adam(model.parameters(), lr= config['lr'], amsgrad= True, weight_decay= 5e-6)
criterion   = torch.nn.CrossEntropyLoss(reduction='none')
scaler      = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience = 50, verbose = True, factor = 0.5)


def compute_bleu_score(dataset):
    candidate_corpus = []
    reference_corpus = []
    size = 0
    batch_bar = tqdm(total=min(20,len(dataset)), dynamic_ncols=True, leave=False, position=0, desc='Train')
    for x, y, lx, ly in dataset:

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly
        size += 1
        predictions, _ = model(x, lx, y)

        predictions = torch.argmax(predictions, axis=2).detach().cpu()
        for batch_idx in range(predictions.shape[0]):

            pred_sliced = indices_to_chars(predictions[batch_idx], VOCAB)

            pred_string = ''.join(pred_sliced)
            candidate_corpus.append(pred_string.split())

            pred_sliced = indices_to_chars(y[batch_idx], VOCAB)

            pred_string = ''.join(pred_sliced)
            reference_corpus.append([pred_string.split()])

        batch_bar.set_postfix(
            score="{:.04f}".format(bleu_score(candidate_corpus=candidate_corpus, references_corpus=reference_corpus)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()
        if size > 20:
          break
    batch_bar.close()

    return bleu_score(candidate_corpus=candidate_corpus, references_corpus=reference_corpus)

compute_bleu_score(test_dataloader)

def train(model, dataloader, criterion, optimizer, teacher_forcing_rate):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    running_perplexity  = 0.0

    for k, (x, y, lx, ly) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.cuda.amp.autocast():

            predictions, attention_plot = model(x, lx, y= y, tf_rate = teacher_forcing_rate)

            loss        =  criterion(predictions.view(-1, predictions.shape[2]), y.reshape(-1))

            mask        = torch.zeros(y.shape).to(DEVICE)
            for i, l in enumerate(ly):
              mask[i,:l] = 1
            masked_loss = torch.sum(loss * mask.reshape(-1)) / torch.sum(mask)
            perplexity  = torch.exp(masked_loss) 

            running_loss        += masked_loss.item()
            running_perplexity  += perplexity.item()

        scaler.scale(masked_loss).backward()

        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(k+1)),
            perplexity="{:.04f}".format(running_perplexity/(k+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_rate))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot

def validate(model, dataloader):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.inference_mode():
            predictions, attentions = model(x, lx, y = None)

        greedy_predictions   =  torch.argmax(predictions, axis=2).detach().cpu()

        running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) 

        batch_bar.set_postfix(
            dist="{:.04f}".format(running_lev_dist/(i+1)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist

config['epochs'] = 200

best_lev_dist = float("inf")
tf_rate = 1.0

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))


    loss, perplexity, attention_plot = train(model, train_dataloader, criterion, optimizer, 0.75)
    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_dist = compute_bleu_score(train_dataloader)
    valid_dist = compute_bleu_score(test_dataloader)
    print("\nEpoch {}/{}: \n Train Loss {:.04f} {:.04f}\t Learning Rate {:.04f}\t Train Distance {:.04f}\t Test Distance {:.04f}".format(
    epoch + 1,
    config['epochs'],
    loss,
    perplexity,
    curr_lr,
    train_dist,
    valid_dist))
    scheduler.step(valid_dist)

    if valid_dist <= best_lev_dist:
      print("Saving model")
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'val_dist': valid_dist,
                  'epoch': epoch}, '/content/drive/MyDrive/Colab Notebooks/11-485/HW4/checkpoint.pth')
      best_val_dist = valid_dist


torch.cuda.empty_cache()
def make_output(model, data_loader):
    model.eval()
    res = []
    true = []
    for x, y, x_lens, y_lens in tqdm(data_loader, position=0, leave=True):
        x = x.to(DEVICE)
        with torch.inference_mode():
            predictions, _ = model(x, x_lens, y=None)
        predictions = torch.argmax(predictions, axis=2).detach().cpu()
        for batch_idx in range(predictions.shape[0]):

            pred_sliced = indices_to_chars(predictions[batch_idx], VOCAB)

            pred_string = ''.join(pred_sliced)
            res.append(pred_string)

            pred_sliced = indices_to_chars(y[batch_idx], VOCAB)

            pred_string = ''.join(pred_sliced)
            true.append(pred_string)

        del x, y, x_lens, y_lens
        torch.cuda.empty_cache()
    return res, true

test_res, true_res = make_output(model, train_dataloader)


for a, b, c, d in test_dataloader:
  print(b)
  break

for i in range(10):
  print(str(i)+ ',' + test_res[i] + "\n")