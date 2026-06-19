import torch 
import torch.nn as nn 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path 
from torch.utils import Dataset,Dataloader, random_split
from dataset import BilingualDataset,causal_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm 


# DATASETS IS USED TO LOAD THE DATASETS 
# TOKENIZER MODEL USED HERE IS 
# THE WORD LEVEL TOKENIZER 
# WHICH CREATES TOKEN FOR EVERYWORD USING WHITESPACES

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}--{config["lang_tgt"]}',split = 'train')
    #building tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])

    #training and validation split
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])
    # this allows the dataset to be splited
    # acc to the size

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    max_len_src = 0
    max_len_src = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    print(f"max length of source sentence : {max_len_src}")
    print(f"max length of target sentence : {max_len_tgt}")

    train_dataloader = Dataloader(train_ds,batch_size = config['batch_size'],shuffle =True)
    val_dataloader = Dataloader(train_ds,batch_size = 1,shuffle =True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def train_model(config):
    #define the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device : {device}")


    Path(config['model_folder']).mkdir(parents =True,exist_ok =True)
    #this checks whether the parent folder(weights) is present or not 
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    #tensorboard 
    # this is to visualise the load :
    writer = SummaryWriter(['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'],eps = 1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # label_smoothing tells to take every 0.1 probabilty fromt the predicted token 
    # to the other tokens 
    # this is done to overfeed
    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc = f"processing epoch {epoch: 02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) #(B,seq_len)
            decoder_input = batch['decoder_input'].to(device) #(B,seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(B,1,seq_len,seq_len)

            #run the tensors through the transformer
            encoder_output = model.encode(encoder_input,encoder_mask) #(b,seq_len,d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) #(b,seq_len,d_model)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device) #(b,seq_len)

            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            #log the loss 
            writer.add_scalar('train_loss',loss.item(),global_step)
            writer.flush()

            #backpropagation
            loss.backward()

            #upadte the weights
            optimizer.step()
            optimizer.zero_grad()
        #save the model at the end of every epoch
        model_filename = get_weights_file_path(config,f"{epoch:02d}")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict,
                'optimizer_state_dict' : optimizer.state_dict(),
                'global_step': global_step
            },model_filename
        )

if __name__ == '__main__':
    config = get_config()
    train_model(config)
    





