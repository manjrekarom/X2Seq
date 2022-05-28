import os
import time
import pickle
import random
import logging
import argparse
from collections import Counter, defaultdict

import spacy
from datasets import load_dataset, load_metric

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from Seq2Vec.model import TextClassifier


parser = argparse.ArgumentParser(description='GRU/LSTM on Document Classification')
parser.add_argument('--vocab-size', '-n', type=int, default=50000, help='Size of vocabulary')
parser.add_argument('--embedding-size', '-e', type=int, default=256, help='Embedding Size')
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden Size of LSTM/GRU')
parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch Size')
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs to train for')
args = parser.parse_args()
print('Training args:', args)


# https://stackoverflow.com/a/56144390/1725038
def default_configuration(logger, level=logging.DEBUG):
    # create console handler and set level to debug
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # # add formatter to ch
    ch.setFormatter(formatter)
    # # add ch to logger
    logger.addHandler(ch)


logger = logging.getLogger('pkg1.pkg2.module')
default_configuration(logger, level=logging.DEBUG)

nlp = spacy.load('en_core_web_sm')
def tokenize(text):
    doc = nlp(text)
    return [tok.text for tok in doc]

train_dataset = load_dataset('emotion', split='train')
val_dataset = load_dataset('emotion', split='validation')
n_classes = 6
# val_dataset = load_dataset('emotion', split='validation+test')

logger.info(f'Train features: {train_dataset}')
logger.info(f'Val features: {val_dataset}')

# sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
label2id = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
id2label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
rand_idx = random.randint(0, train_dataset.num_rows-1)
logger.debug(f'An example: {train_dataset[rand_idx]["text"]}, Label: {id2label[train_dataset[rand_idx]["label"]]}')


# build a vocabulary
if os.path.exists('vocab.pkl'):
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
else:
    build_vocab_start = time.time()
    vocab = Counter()
    for i in range(train_dataset.num_rows):
        text = train_dataset[i]['text']
        tokens = tokenize(text)
        vocab.update(tokens)
        if i <= 5:
            logger.debug(f'{tokens}')
            logger.debug(f'Vocab: {vocab}')
    build_vocab_time = time.time() - build_vocab_start
    logger.info(f'Building vocab took: {build_vocab_time}s')
    # save vocab in pickle
    with open('vocab.pkl', 'wb+') as f:
        pickle.dump(vocab, f)

logger.info(f'No. of vocabulary words: {len(vocab)}')
vocab_counts = vocab.most_common(args.vocab_size)
del vocab
# don't need counts anymore
id2vocab, _ = list(zip(*vocab_counts))
id2vocab = list(id2vocab)
# add special tokens
pad_token, pad_idx = '<pad>', 0
unk_token, unk_idx = '<unk>', 1
eos_token, eos_idx = '<eos>', 2
id2vocab.insert(pad_idx, pad_token)
id2vocab.insert(unk_idx, unk_token)
id2vocab.insert(eos_idx, eos_token)
logger.debug(f'First four vocab: {id2vocab[:4]}')

vocab2id = {id2vocab[i]: i for i in range(len(id2vocab))}
logger.debug(f'vocab2id: {vocab2id[id2vocab[0]]}, {vocab2id[id2vocab[-1]]}')
logger.debug(f'id2vocab: {id2vocab[0]}, {id2vocab[-1]}')
vocab_size = len(id2vocab)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# model
text_classifier = TextClassifier(vocab_size, args.embedding_size, args.hidden_size, n_classes).to(device)

# criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(text_classifier.parameters(), lr=args.lr)


def convert_to_ids(text):
    ids_text = []
    tokenized_text = tokenize(text)
    # convert to ids
    for token in tokenized_text:
        if token in vocab2id:
            ids_text.append(vocab2id[token])
        else:
            # handle out of vocab tokens
            ids_text.append(unk_idx)
    # add eos token
    ids_text.append(eos_idx)
    return torch.LongTensor(ids_text)


# other metrics
train_loss = 0.0
val_loss = 0.0

# train and val loops
for i in range(args.num_epochs):
    # initialize metrics for epochs
    train_acc = load_metric('accuracy')
    val_acc = load_metric('accuracy')
    train_f1_score = load_metric('f1')
    val_f1_score = load_metric('f1')

    print(f'Epoch {i+1}')
    print('=' * 40)

    # train
    text_classifier.train()
    for batch_i, batch in enumerate(train_loader):
        # preprocess batch
        batch_text = []
        batch_label = torch.LongTensor(batch['label']).to(device)
        
        # tokenize and indexify text
        for text_item_i, text_item in enumerate(batch['text']):
            if batch_i == 0 and text_item_i == 0:
                logger.debug(f'Converting "{text_item}" to ids: {convert_to_ids(text_item)}')
            batch_text.append(convert_to_ids(text_item))
        
        # note sequence lengths
        seq_lens = [text.shape[0] for text in batch_text]
        batch_text = pad_sequence(batch_text, batch_first=True, padding_value=pad_idx).to(device)
        if batch_i == 0:
            logger.debug(f'batch_text: {batch_text}')
            logger.debug(f'Shape of batch_text: {batch_text.shape}')
        assert len(batch_text) == batch_label.shape[0]
        
        # model forward
        logits = text_classifier(batch_text, seq_lens)
        preds = torch.argmax(logits, dim=-1)
        loss = criterion(logits, batch_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = ((batch_i * train_loss) + loss.item()) / (batch_i + 1)
        train_acc.add_batch(predictions=preds, references=batch_label)
        train_f1_score.add_batch(predictions=preds, references=batch_label)
        # if (batch_i + 1) % 100 == 0:
        #     print(f'Batch: {batch_i + 1}, Train Loss: {train_loss}, Train Acc: {train_acc.compute()}, \
        #         Train F1: {train_f1_score.compute(average="macro")}')

    print(f'Train Loss: {train_loss}, Train Acc: {train_acc.compute()}, Train F1: {train_f1_score.compute(average="macro")}')

    # eval
    text_classifier.eval()
    for batch_i, batch in enumerate(val_loader):
        # preprocess batch
        batch_text = []
        batch_label = torch.LongTensor(batch['label']).to(device)

        # tokenize and indexify text
        for text_item_i, text_item in enumerate(batch['text']):
            if batch_i == 0 and text_item_i == 0:
                logger.debug(f'Converting "{text_item}" to ids: {convert_to_ids(text_item)}')
            batch_text.append(convert_to_ids(text_item))

        # note sequence lengths
        seq_lens = [text.shape[0] for text in batch_text]
        batch_text = pad_sequence(batch_text, batch_first=True, padding_value=pad_idx).to(device)
        assert len(batch_text) == batch_label.shape[0]
        
        # model forward
        with torch.no_grad():
            logits = text_classifier(batch_text, seq_lens)
            loss = criterion(logits, batch_label)
        preds = torch.argmax(logits, dim=-1)

        val_loss = ((batch_i * val_loss) + loss.item()) / (batch_i + 1)
        val_acc.add_batch(predictions=preds, references=batch_label)
        val_f1_score.add_batch(predictions=preds, references=batch_label)

    print(f'Val Loss: {val_loss}, Val Acc: {val_acc.compute()}, Val F1: {val_f1_score.compute(average="macro")}')
    torch.save(text_classifier.state_dict(), 'model.pt')
