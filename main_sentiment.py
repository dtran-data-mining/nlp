#!/usr/bin/env python
# coding: utf-8
'''fun: sentiment analysis'''


import torch
from torch import nn
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import time


'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('device: ', device)

    ## adjust parameter settings
    mode = 'train'
    batch_size = 300
    ## choose 1-3 layers
    n_layers = 1 

    ## input seq length aligned with data pre-processing
    input_len = 150

    ## word embedding length
    embedding_dim = 200

    # lstm hidden dim
    hidden_dim = 50
    # binary cross entropy
    output_size = 1
    num_epoches = 1
    global_step = 0
    ## learning rate
    learning_rate = 0.002
    # gradient clipping
    clip = 5
    load_cpt = False #True
    # ckp_path = 'cpt/name.pt'
    ## use pre-train Glove embedding or not?
    pretrain = True

    ##-----------------------------------------------------------------------
    ## Bonus (5%): complete code to add GloVe embedding file path below.
    ## Download Glove embedding from https://nlp.stanford.edu/data/glove.6B.zip
    ## "embedding_dim" defined above shoud be aligned with the dimension of GloVe embedddings
    ##-----------------------------------------------------------------------
    glove_file = 'glove.6B/glove.6B.200d.txt'
    
    ##  load training and test data from data loader
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=batch_size,\
                                    shuffle=True,num_workers=1)
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=batch_size,\
                                shuffle=False,num_workers=1)


    ## [Bonus] read tokens and load pre-train embedding
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    ## -----------------------------------------------
    ## import model from LSTM.py
    ## complete the code in "def forward(self, x)" in LSTM.py file
    ## then import model from LSTM.py below
    ## -----------------------------------------------
    model = LSTMModel(vocab_size, output_size, embedding_dim, embedding_matrix,
                 hidden_dim, n_layers, input_len, pretrain)
    model.to(device)

    ## define optimizer and loss function
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()
    
    ## load checkpoint
    if load_cpt:
        print("*"*10+'loading checkpoint'+'*'*10)
        ##-----------------------------------------------   
        ## complete code below to load checkpoint
        ##-----------------------------------------------
        


    ## model training
    print('*'*89)
    print('start model training now')
    print('*'*89)
    if mode == 'train':
        model.train()
        for epoch in range(num_epoches):
            for x_batch, y_labels in training_generator:
                global_step += 1

                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                ## predict result from model
                y_out = model(x_batch)

                ## compute loss
                loss = criterion(y_out, y_labels)

                ## back propagation
                optimizer.zero_grad()
                loss.backward()
                ## clip_grad_norm helps prevent the exploding gradient problem in LSTMs
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            
            ## save checkpoint
            ckp_path = f'checkpoint/step_{global_step}.pt'
            _save_checkpoint(ckp_path, model, epoch, global_step, optimizer)
    
    ## model testing
    print("----model testing now----")
    model.eval()
    with torch.no_grad():
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            # predict result is a single value between 0 and 1
            y_out = model(x_batch)
            # round predicted label to 0 or 1
            y_pred = torch.round(y_out) 
    


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")

    