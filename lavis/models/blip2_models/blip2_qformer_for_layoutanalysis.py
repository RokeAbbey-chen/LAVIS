import torch
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.common.registry import registry
from transformers import BertTokenizer

@registry.register_model("blip2")
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


    def itg(self, text_tokens, query_tokens, image, query_output, samples):
        best_output = None

        for i, text in enumerate(samples['all_text_input']):
            
            text_tokens = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
        
            lm_output = super(Blip2QformerForLayoutAnalysis, self).itg(text_tokens, query_tokens, image, query_output, samples)
            if best_output is None:
                best_output = lm_output
            elif best_output.loss > lm_output.loss:
                best_output = lm_output
        # loss_lm = lm_output.loss
        return best_output 