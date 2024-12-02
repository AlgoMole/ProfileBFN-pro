import os
import re
import glob
from typing import Optional
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union

from transformers.models.esm.modeling_esm import *
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


from fairseq.models.roberta.bfn_roberta import BFNRobertaEncoder
from fairseq.models.roberta.p_roberta import ESM2RobertaEncoder
from fairseq.data import Dictionary
from fairseq.data.bfn4seq_dataset import (
    _tensor_discreteBayesianFlow_mbcltbf,
    _np_discreteBayesianFlow_mbcltbf,
    _np_discreteBayesianFlow_mnbf,
)

def upgrade_state_dict(state_dict):
    """Removes prefixes 'encoder.sentence_encoder.' and
    'encoder.'"""
    prefixes = ["encoder."]
    pattern = re.compile("^" + "|".join(prefixes))  # ^encoder.sentence_encoder|encoder
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict



class EsmForBFN(nn.Module):
    def __init__(self, ckpt_path):
        super(EsmForBFN, self).__init__()

        self.model_data = torch.load(ckpt_path, map_location="cpu")
        dict_path = f'{glob.glob(self.model_data["cfg"]["task"]["data"])[0]}/dict.txt'
        self.dictionary = Dictionary.load(dict_path)
        self.dictionary.add_symbol("<null_1>")
        self.dictionary.add_symbol("<mask>")

        self.mask_id = self.dictionary.index("<mask>")
        self.bos_id = self.dictionary.index("<s>")
        self.eos_id = self.dictionary.index("</s>")
        self.pad_id = self.dictionary.index("<pad>")
        self.x_id = self.dictionary.index("X")

        print(len(self.dictionary.symbols), self.dictionary.symbols)
        cfg = self.model_data["cfg"]["model"]
        state_dict = upgrade_state_dict(self.model_data["model"])
        encoder = BFNRobertaEncoder(args=cfg, dictionary=self.dictionary).cuda()
        self.hidden_size = encoder.hidden_size
        assert sorted(encoder.state_dict().keys()) == sorted(state_dict.keys())
        encoder.load_state_dict(state_dict)
        encoder.eval()
        self.bfn_encoder = encoder
    
    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):
        
        # print(f"input_ids [{input_ids.shape}] = {input_ids}")

        B, T = input_ids.size()
        src_lengths = torch.tensor([T], dtype=torch.long, device=input_ids.device)

        # find the masked pos and turn it to uniform
        be_masked = (input_ids == self.mask_id)
        # print(f"be_masked [{be_masked.shape}] = {be_masked}")
        uniform_input_matrix = torch.randint(0, 20, (B, T), dtype=torch.long, device=input_ids.device) + 4
        input_ids[be_masked] = uniform_input_matrix[be_masked]


        loop_probs = (
            F.one_hot(input_ids, num_classes=len(self.dictionary)).float().to(input_ids.device)
        )


    
        t = torch.ones([B, T], dtype=torch.float32, device=input_ids.device)

        beta1 = torch.ones_like(t) * self.model_data["cfg"]["model"].beta1
        bfn_tokens = _tensor_discreteBayesianFlow_mbcltbf(
            t,
            input_ids,
            beta1,
            dict_size=len(self.dictionary),
            torder=self.model_data["cfg"]["model"].beta_time_order,
        )
        pred_tokens, extra = self.bfn_encoder.forward(t, bfn_tokens, src_lengths, return_last_hidden=True)    # [B, T, V]
        # print(f"pred_tokens.shape = {pred_tokens.shape}")
        
        V = pred_tokens.shape[-1]
        # pred_tokens[be_masked] = torch.ones([V], dtype=torch.float32, device=input_ids.device) * 1.0 / V

        result = {
            "logits": pred_tokens,
            "last_hidden_state": extra,
        }
        return result
    
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    # def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
    #     non_special_sym_mask = (
    #         output_tokens.ne(self.pad_id) &
    #         output_tokens.ne(self.bos_id) &
    #         output_tokens.ne(self.eos_id)
    #     )
    #     if partial_masks is not None:
    #         non_special_sym_mask &= (~partial_masks)
    #     return non_special_sym_mask
    
    # def initialize_output_tokens(self, batch, encoder_out, partial_masks=None, **kwargs):
    #     tokens = batch['prev_tokens']
    #     if tokens is None:
    #         raise NotImplementedError
    #     else:
    #         output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

    #         output_tokens = tokens.masked_fill(output_mask, self.mask_id)
    #         output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

    #         return output_tokens, output_scores