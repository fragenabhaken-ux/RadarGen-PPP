#!/usr/bin/env python3
"""Prepare the baseline Gemma empty prompt from explicitly supplied local assets."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-model-dir', required=True)
    parser.add_argument('--output', default='output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not Path(args.text_model_dir).is_dir():
        parser.error('--text-model-dir must contain the local Efficient-Large-Model/gemma-2-2b-it assets')
    if Path(args.output).exists():
        parser.error('output already exists; inspect and reuse the existing embedding')
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True)
    tokenizer.padding_side = 'right'
    encoder = AutoModelForCausalLM.from_pretrained(args.text_model_dir, local_files_only=True,
                                                torch_dtype=torch.bfloat16).get_decoder().to(args.device).eval()
    tokens = tokenizer('', max_length=300, padding='max_length', truncation=True,
                       return_tensors='pt').to(args.device)
    with torch.no_grad():
        embedding = encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    if embedding.shape != (1,300,2304) or not torch.isfinite(embedding).all():
        raise ValueError('Local text model does not produce the expected finite Gemma embedding')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'uncond_prompt_embeds':embedding.cpu(),
                'uncond_prompt_embeds_mask':tokens.attention_mask.cpu()},args.output)
    print(args.output)

if __name__ == '__main__':
    main()
