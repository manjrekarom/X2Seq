import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, classes):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.classifier_head = nn.Linear(hidden_size, classes)

    def forward(self, inputs, seq_lens):
        embeddings = self.word_embeddings(inputs)
        packed_embeddings = pack_padded_sequence(embeddings, seq_lens, batch_first=True,
        enforce_sorted=False)
        last_layer_hidden, (hn, cn) = self.lstm(packed_embeddings)
        last_layer_hidden, seq_lens_ = pad_packed_sequence(last_layer_hidden, batch_first=True, padding_value=0)
        # print(f'Passed seq_lens: {seq_lens}')
        # print(f'Returned seq_lens: {seq_lens_}')
        # print(f'last layer hidden shape: {hn.shape}')
        batch_size = last_layer_hidden.size(0)
        indices = torch.LongTensor(seq_lens) - 1
        hn_eos = last_layer_hidden[torch.arange(batch_size), indices, ...]
        # print(f'hn_eos shape: {hn_eos.shape}')
        # print(f'hn shape: {hn.shape}')
        # first hidden state
        # print(hiddens[:, 0, ...].squeeze(dim=1).shape)
        # return self.classifier_head(hn[:, 0, ...].squeeze(dim=1))
        return self.classifier_head(hn_eos)


if __name__ == "__main__":
    vs, es, hs, c = 10, 3, 4, 2
    tc = TextClassifier(vs, es, hs, c)
    bs, seq_len = 5, 6
    print(tc(torch.randint(0, vs, (bs, seq_len))))
