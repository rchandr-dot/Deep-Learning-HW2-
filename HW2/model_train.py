import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset


# Function to preprocess data and create word dictionaries
def data_preprocessing():
    filepath = 'MLDS_hw2_1_data/'
    with open(filepath + 'training_label.json', 'r') as f:
        file = json.load(f)
    # Counting word occurrences
    word_count = {}
    for item in file:
        for sent in item['caption']:
            word_sentence = re.sub('[.!,;?]', ' ', sent).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    # Create dictionaries for word-to-index and index-to-word mappings
    # here we are using the tokens mentioned in the hwppt
    word_dict = {}
    for word in word_count:
        if word_count[word] > 4:
            word_dict[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, word_dict
# Function to split and preprocess a sentence
def s_split(sentence, word_dict, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)  # <SOS> token
    sentence.append(2)     # <EOS> token
    return sentence
# Function to annotate captions
def annotate(label_file, word_dict, w2i):
    label_json = 'MLDS_hw2_1_data/' + label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, word_dict, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption
# Function to load audio-visual features
def avi(files_dir):
    avi_data = {}
    training_feats = 'MLDS_hw2_1_data/' + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data
# Function to create minibatches
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths
# Dataset class for training data
class training_data(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = avi(label_file)
        self.w2i = w2i
        self.data_pair = annotate(files_dir, word_dict, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000.
        return torch.Tensor(data), torch.Tensor(sentence)

# Dataset class for testing data
class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        return self.avi[idx]

# Attention mechanism class
class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()

        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context

# Encoder class
class encoderRNN(nn.Module):
    def __init__(self):
        super(encoderRNN, self).__init__()

        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, hidden_state = self.gru(input)

        return output, hidden_state

# Decoder class
class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.5):
        super(decoderRNN, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()

        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len - 1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold:
                current_input_word = targets[:, i]
            else:
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28

        for i in range(assumption_seq_len - 1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return expit(training_steps / 20 + 0.85)
# Combined model class
class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(
                encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs,
                targets=target_sentences, mode=mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state,
                                                              encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions
# Function to calculate loss
def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss / batch_size

    return loss
# Function to train the model
def train(model, epoch, loss_fn, parameters, optimizer, train_loader, scheduler=None):
    model.train()
    print(epoch)

    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)

        loss.backward()
        optimizer.step()

    if scheduler:
        scheduler.step()

    loss = loss.item()
    print(loss)
# Function to test the model
def test(test_loader, model, i2w):
    model.eval()
    ss = []
    for batch_idx, batch in enumerate(test_loader):
        id, avi_feats = batch
        avi_feats = avi_feats.cuda()
        id, avi_feats = id, Variable(avi_feats).float()

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions

        result = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(id, result)
        for r in rr:
            ss.append(r)
    return ss

# Main function
def main():
    i2w, w2i, word_dict = data_preprocessing()
    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(i2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

    label_file = 'training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = training_data(label_file, files_dir, word_dict, w2i)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)

    epochs_n = 100

    encoder = encoderRNN()
    decoder = decoderRNN(512, len(i2w) + 4, len(i2w) + 4, 1024, dropout_percentage=0.5)
    model = MODELS(encoder=encoder, decoder=decoder)

    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)

    # Add learning rate scheduler
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs_n):
        train(model, epoch + 1, loss_fn, parameters, optimizer, train_dataloader, scheduler=scheduler)

    torch.save(model, "{}/{}.h5".format('SavedModel', 'model1'))
    print("Training finished")

if __name__ == "__main__":
    main()
