import pickle
import argparse
from collections import Counter, defaultdict

import spacy
from datasets import load_dataset, load_metric

import torch
from torch.nn.utils.rnn import pad_sequence

from Seq2Vec.model import TextClassifier


parser = argparse.ArgumentParser(description='GRU/LSTM on Document Classification')
parser.add_argument('example', type=str, help='Example to run the prediction on')
parser.add_argument('--vocab-size', '-n', type=int, default=50000, help='Size of vocabulary')
parser.add_argument('--embedding-size', '-e', type=int, default=256, help='Embedding Size')
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden Size of LSTM/GRU')
args = parser.parse_args()
print('Prediction args:', args)

n_classes = 6

# sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
label2id = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
id2label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# load vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

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

vocab2id = {id2vocab[i]: i for i in range(len(id2vocab))}
vocab_size = len(id2vocab)

nlp = spacy.load('en_core_web_sm')
def tokenize(text):
    doc = nlp(text)
    return [tok.text for tok in doc]

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

text_classifier = TextClassifier(vocab_size, args.embedding_size, args.hidden_size, n_classes)
text_classifier.load_state_dict(torch.load('model.pt'))

batch_text = convert_to_ids(args.example).unsqueeze(0)
seq_lens = [text.shape[0] for text in batch_text]
with torch.no_grad():
    logits = text_classifier(batch_text, seq_lens)
preds = torch.argmax(logits, dim=-1)
print(id2label[preds.item()])
