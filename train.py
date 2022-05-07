import argparse
from typing import Iterable

from ipywidgets import Output
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.auto as tqd

from source.generator import Generator, ArgmaxGenerator, \
    DistributionGenerator, BeamSearchGenerator
from source.model import RNNModel
from source.utils import Lang, PAD_TOKEN_INDEX
from source.train_utils import NewsDataset, padding_collater, \
    SequenceNLLLoss, split_dataset


def train(model, 
          device,
          generators: Iterable[Generator], 
          loss, opt, 
          train_dataloader: DataLoader, val_dataloader: DataLoader, 
          n_epochs: int,
          visualize=False):
    logs = {}
    logs['loss'] = []
    logs['val_loss'] = []
    
    out = Output()
    display(out)
    
    for i in range(n_epochs):
        print(f'Epoch {i + 1}')

        # train
        epoch_train_loss = []
        for batch_index, batch in tqd.tqdm(enumerate(train_dataloader)):
            batch = batch.to(device)
            opt.zero_grad()
            loss_value = loss(model, batch)
            loss_value.backward()
            opt.step()

            epoch_train_loss.append(loss_value.item())
            
        logs['loss'].extend(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = []
        for batch_index, batch in tqd.tqdm(enumerate(val_dataloader)):
            batch = batch.to(device)
            with torch.no_grad():
                loss_value = loss(model, batch)
            epoch_val_loss.append(loss_value.item())
        
        logs['val_loss'].append(np.mean(epoch_val_loss))
            
        with out:
            out.clear_output(wait=True)
            print('Train loss:', np.mean(epoch_train_loss))
            print('Val loss:', np.mean(epoch_val_loss))
            for gen in generators:
                print(gen.name())
                for j in range(3):
                    print(gen.generate(model, device))
            
            if visualize:
                fig2 = plt.figure(figsize=(12, 6))
                for i, (name, los) in enumerate(logs.items()):
                    plt.subplot(1, 2, i+1)
                    plt.plot(los)
                    plt.title(name)
                    plt.show(fig2)
            
        model.train()

        
def main(train_data_path: str,
         batch_size: int,
         embedding_dim: int,
         hidden_dim: int,
         device_name: str,
         n_epochs: int,
         checkpoint_path: str):
    with open(train_data_path, 'r') as f_in:
        news_sentences_bpe = f_in.readlines()
            
    bpe_lang = Lang(tokenizer=lambda x: x.split(), lemmatizer=lambda x: x)
    bpe_lang.addDocument(news_sentences_bpe)
    bpe_lang.getStat()
    NUM_EMBEDDINGS = len(bpe_lang.index2word)
        
    empty_emb_layer = nn.Embedding(num_embeddings=NUM_EMBEDDINGS,
                                   embedding_dim=embedding_dim)
    
    device = torch.device('cpu')
    if device_name == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA device successfuly initialized")
        else:
            print("CUDA device unavailable; using CPU")
    else:
        print("Using CPU device")
    
        
    model = RNNModel(empty_emb_layer, 
                     hidden_dim=hidden_dim).to(device)
    loss = SequenceNLLLoss(ignore_index=PAD_TOKEN_INDEX)
    
    gens = [ArgmaxGenerator(bpe_lang.index2word, max_length=100), 
            DistributionGenerator(bpe_lang.index2word, max_length=100, k_max=20)]
    
    bpe_dataset = NewsDataset(
        sentences=news_sentences_bpe,
        tokenizer=bpe_lang.tokenizer,
        lemmatizer=bpe_lang.lemmatizer,
        word2index=bpe_lang.word2index
    )
    bpe_dataset_train, bpe_dataset_val = split_dataset(bpe_dataset)

    train_dataloader = DataLoader(bpe_dataset_train, batch_size=batch_size, shuffle=True, 
                                  collate_fn=padding_collater)
    val_dataloader = DataLoader(bpe_dataset_val, batch_size=batch_size, shuffle=False, 
                                collate_fn=padding_collater)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, device, gens, loss, opt, train_dataloader, val_dataloader, n_epochs)
    
    torch.save(model.state_dict(), path)
    

parser = argparse.ArgumentParser(description='Train RNN model for generation.')
parser.add_argument('--data-path', type=str, default='data/train.bpe.ru',
                    help='path to file train sentences')
parser.add_argument('--embedding-dim', default=256,
                    help='embedding size')
parser.add_argument('--hidden-dim', default=1024,
                    help='hidden state size')
parser.add_argument('--device-name', default='cuda',
                    help='device name. to use cpu for training change value to any')
parser.add_argument('--n-epochs', default=3)
parser.add_argument('--batch-size', default=32)
parser.add_argument('--checkpoint-path', default='pretrained_models/bpe_4000_1024',
                    help='path to save trained model')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_path,
         args.batch_size,
         args.embedding_dim,
         args.hidden_dim,
         args.device_name,
         args.n_epochs,
         args.checkpoint_path)
