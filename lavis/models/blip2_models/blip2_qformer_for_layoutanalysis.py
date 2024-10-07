from typing import Any, Mapping
import torch
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.common.registry import registry
from transformers import BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

@registry.register_model("blip2_for_layoutananlysis")
class Blip2QformerForLayoutAnalysis(Blip2Qformer):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        delimiter = "<del>"
        if delimiter not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [delimiter]})

        return tokenizer
    
    
    # def forward


    def itg(self, text_tokens, query_tokens, image, query_output, samples, reduction='none'):
        best_output = None
        reduction = 'none'


        
        # best_output = BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=None,
        #     pooler_output=None,
        #     past_key_values=None,
        #     hidden_states=None,
        #     attentions=None,
        #     cross_attentions=None
        # )

        for c in range(len(samples['all_text_input'][0])):
            texts_c = []
            for r, text in enumerate(samples['all_text_input']):
                texts_c.append(text[c])
            text_tokens = self.tokenizer(
                    texts_c,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

            lm_output = super(Blip2QformerForLayoutAnalysis, self).itg(text_tokens, query_tokens, image, query_output, samples, reduction=reduction)
            if 0 == c:
                best_output = lm_output
            else:
                mask = best_output.loss > lm_output.loss
                best_output.loss[mask] = lm_output.loss[mask]
                best_output.logits[mask] = lm_output.logits[mask]
                best_output.past_key_values = None
                best_output.hidden_states = None
                best_output.attentions = None
                best_output.cross_attentions = None
        # loss_lm = lm_output.loss
        best_output.loss = best_output.loss.mean()
        return best_output 

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        embedding_keys = ["Qformer.bert.embeddings.word_embeddings.weight",
                          "Qformer.cls.predictions.bias",
                          "Qformer.cls.predictions.decoder.weight",
                          "Qformer.cls.predictions.decoder.bias"]
        self_state_dict = self.state_dict()
        for k in embedding_keys:
            if -1 <= self_state_dict[k].shape[0] - state_dict[k].shape[0] <= 1:
                print("delete key:", k)
                del state_dict[k]

        return super().load_state_dict(state_dict, strict, assign)