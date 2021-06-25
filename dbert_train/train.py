import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

from transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from model_distil import DistilBertForEncoderMLM
from utils import (
    init_logger,
    set_seed
)
from data_loader_distil import (
    load_and_cache_examples
)

logger = logging.getLogger(__name__)
from itertools import cycle


from transformers import PreTrainedTokenizer
from typing import Tuple

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels



def train(args,
          model,
          tokenizer,
          train_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    loss_list = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
#             batch = tuple(t.to(args.device) for t in batch)

            temp_inputs, temp_masked_labels = mask_tokens(batch[0], tokenizer, args)
            temp_inputs = temp_inputs.to(args.device)
            temp_masked_labels = temp_masked_labels.to(args.device)
            
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": temp_inputs, 
                "attention_mask": batch[1],
#                 "token_type_ids": batch[2],
                "labels": temp_masked_labels
            }
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
#             print(loss)
#             print(loss.item())
            k = loss.item()
            loss_list.append(k)
#             print(loss_list)
#             import sys
#             sys.exit()
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
#                 if args.max_steps > 0 and global_step > args.max_steps:
#                     break
                #if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
              
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to {}".format(output_dir))

        if args.save_optimizer:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            
        output_eval_file = os.path.join(output_dir, "loss-{}.txt".format(global_step))
        print("Losses:")
        print(loss_list)
        with open(output_eval_file, "w") as f_w:
            for i in loss_list:
                print(i,file=f_w)
            
        if args.max_steps > 0 and global_step > args.max_steps:
            break
    print("Losses:  ")
    print(loss_list)
    output_eval_file = os.path.join(output_dir, "loss.txt")
    with open(output_eval_file, "w") as f_w:
        for i in loss_list:
            print(i,file=f_w)
    return global_step, tr_loss / global_step

def main(cli_args):
    # Read from config file and make args
    config_filename = "enc_mlm_distil.json"
    with open(config_filename) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    args.mlm_probability=0.15
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)


    config = DistilBertConfig.from_pretrained(
        args.model_name_or_path,
        finetuning_task=args.task
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )

    model = DistilBertForEncoderMLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    special_tokens_dict = {'additional_special_tokens': ['[STATE]','[ACTION]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
#     checkpoint = "/content/gdrive/MyDrive/distil-ckpt/distilbert-base-cased/checkpoint-13594"
#     model = DistilBertForEncoderMLM.from_pretrained(checkpoint)
    
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None 
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # cli_parser.add_argument("--taxonomy", type=str, required=True, help="Taxonomy (original, ekman, group)")

    cli_args = cli_parser.parse_args()

    main(cli_args)
