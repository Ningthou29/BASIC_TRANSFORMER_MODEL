from typing import Any

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['SOS'])],dtype = torch.float64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['EOS'])],dtype = torch.float64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['PAD'])],dtype = torch.float64)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self,index : Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        # this is too take out the tokens 

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        # this is convert the tokens into ids 
        dec_input_tokens = self.tokenizer_tgt.encode(src_text).ids
        # same herer as above 

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # this is to create the padding 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        # in the decoder only the sos is added
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError('Sentence is too long')
        #add sos and eos to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,torch.tensor(enc_input_tokens,dtype = torch.int64),self.eos_token,torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype = torch.int64)

            ]
        )
        #only add SOS to the decoder input
        decoder_input = torch.cat(
            [
               self.sos_token,
               torch.tensor(dec_input_tokens,dtype = torch.int64),
               torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype = torch.int64)

            
            ]

        )
        #target

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype = torch.int64)

            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #(Seq_len)
            "decoder_input" : decoder_input, #(Seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,Seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1,Seq_len) & (1,Seq_len,Seq_len)
            " src_text" : src_text,
            "tgt_text": tgt_text
        }
# to mask we build a method named causal_mask
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal =1).type(torch.int)
    return mask == 0

