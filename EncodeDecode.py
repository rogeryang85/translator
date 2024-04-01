# Originally a notebook, but converted it to a python file

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

from google.colab import drive
drive.mount('/content/drive')

class EnFr_Dataset(Dataset):
    def __init__(self, split, language_pair = ('fr', 'en')):
        self.df = None
        if split == 'train':
            self.df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/aa/en-fr.csv', nrows=100000).dropna()
        else:
            self.df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/aa/en-fr.csv', skiprows=100000, nrows=1000).dropna()
      if split == 'train':
            self.df = self.df[self.df.apply(lambda x: len(str(x['en'])) < 150 and len(str(x['fr'])) < 150, axis=1)]
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

Dataset = EnFr_Dataset
Training_Dataset = Dataset(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
Validation_Dataset = Dataset(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

def french_to_english(text):
    english_chars = string.ascii_letters + string.digits + string.punctuation + ' '

    converted = ''
    for char in text:
        if char in english_chars:
            converted += char
        else:
            converted += unidecode.unidecode(char)

    return converted


chars = list("PSEabcdefghijklmnopqrstuvwxyz ")
def one_hot(string):
  # Dictionary mapping each character to an index
  char_to_idx = {ch:i for i,ch in enumerate(chars)}
  # Empty list to hold one-hot encoded arrays
  encoded = []

  for char in string:
      if char not in char_to_idx:
        continue
      # Create a zero array with length equal to number of possible chars
      arr = torch.zeros(len(chars))
      # Set the index corresponding to the current char to 1
      arr[char_to_idx[char]] = 1
      # Add to encoded list
      encoded.append(arr)

  return torch.stack(encoded)

def indices(string):
  # Dictionary mapping each character to an index
  char_to_idx = {ch:i for i,ch in enumerate(chars)}
  # Empty list to hold one-hot encoded arrays
  encoded = []

  for char in string:
      if char not in char_to_idx:
        continue
      # Create a zero array with length equal to number of possible chars
      arr = torch.zeros(1)
      # Set the index corresponding to the current char to 1
      arr[0] = char_to_idx[char]
      # Add to encoded list
      encoded.append(arr)

  return torch.stack(encoded).reshape(-1)

def clean(string):
  char_to_idx = {ch:i for i,ch in enumerate(chars)}
  string = 'S' + string + 'E'
  # Empty list to hold one-hot encoded arrays
  encoded = []
  new_string = ""
  for char in string:
      if char not in char_to_idx:
        continue
      new_string += char

  return new_string

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        cleaned_english = src_sample.rstrip("\n").lower()
        cleaned_english = clean(cleaned_english)
        stripped_tgt = tgt_sample.rstrip("\n")
        cleaned_french = french_to_english(stripped_tgt).lower()
        cleaned_french = clean(cleaned_french)

        src_batch.append(cleaned_english)
        tgt_batch.append(cleaned_french)

    max_len_src = len(max(src_batch, key=len))
    max_len_tgt = len(max(tgt_batch, key=len))
    # Pad all other strings with zeros to match length
    padded_src = [s.ljust(max_len_src, 'E') for s in src_batch]
    padded_tgt = [s.ljust(max_len_tgt, 'E') for s in tgt_batch]

    src_batch_new = []
    tgt_batch_new = []
    for s in padded_src:
      src_batch_new.append(one_hot(s))
    for s in padded_tgt:
      tgt_batch_new.append(indices(s))

    return torch.stack(src_batch_new), torch.stack(tgt_batch_new).long()

train_dataloader = DataLoader(Training_Dataset, batch_size=4, collate_fn=collate_fn)

for example_batch in train_dataloader:
  a, b = example_batch
  print(a.shape)
  print(b.shape)
  break

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden
    
class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=DEVICE).fill_(1)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        if target_tensor is not None:
            lens = target_tensor.shape[1]
        else:
            lens = 1200

        for i in range(lens):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
class ED(torch.nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_output_size):

        super(ED, self).__init__()

        self.encoder        = EncoderRNN(input_size, encoder_hidden_size)
        self.decoder        = DecoderRNN(encoder_hidden_size, decoder_output_size)

    def forward(self, x, y = None, tf_rate = 1):

        encoder_outputs, hidden = self.encoder(x) # from Listener
        predictions, hidden, _ = self.decoder(encoder_outputs, hidden, y)

        return predictions
    
model = ED(30, 256, 30)

model = model.to(DEVICE)
print(model)

summary(model,
        x= example_batch[0].to(DEVICE),
        y= example_batch[1].to(DEVICE))

model(example_batch[0].to(DEVICE), example_batch[1].to(DEVICE)).shape

def train_epoch(dataloader, model, optimizer, criterion):

    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        model.zero_grad()
        decoder_outputs = model(input_tensor.to(DEVICE), target_tensor.to(DEVICE))

        loss = criterion(
            decoder_outputs,
            target_tensor
        )
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train(train_dataloader, model, n_epochs, learning_rate=0.001,
               print_every=1, plot_every=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, model, optimizer, loss_fn)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

train(train_dataloader, model, 2)

def convert(t):
  chars = list("PSEabcdefghijklmnopqrstuvwxyz ")
  t = torch.max(t, axis=2)[1]
  outputs = []
  for i in range(t.shape[0]):
    s = ""
    for j in range(t.shape[1]):
      s += chars[t[i][j]]
    outputs.append(s)
  return outputs

out = model(example_batch[0].to(DEVICE), example_batch[1].to(DEVICE))

convert(out)

convert(example_batch[1])