import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from dataset import essayDataset
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp
import wandb

logging.getLogger().setLevel(logging.CRITICAL)


def define_argparser():
    """
    Define argument parser
    :return: configuration object
    """
    parser = argparse.ArgumentParser(description="run argparser")
    parser.add_argument(
        "--data_path",
        required=False,
        default="'./data",
        help="storyline data path (csv format), must include content, genre columns",
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        default="./trained_models/",
        help="where to save model checkpoint",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="print average loss at every n step",
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="save the model at every N epoch"
    )
    args = parser.parse_args()
    return args

def auto_enter(text):
    text = (text.replace("   ", "\n"))
    text = text.split("\n")
    text = [t.lstrip() for t in text if t != '']
    return "\n\n".join(text)


def generation_byt(input_sentence, model, tok, vocab, temperature=0.8, top_p=0.95, top_k=50, text_size=200):
    ctx = 'cuda:0'
    device = torch.device(ctx)

    sent = ''
    sent = sent + input_sentence
    toked = tok(sent)

    input_ids = torch.tensor([vocab[vocab.bos_token], ] +  vocab[toked]).unsqueeze(0)
    input_ids = input_ids.to(ctx)
    # outputs = self.kogpt2model.generate(input_ids=input_ids, max_length=200, repetition_penalty=1.2, pad_token_id=3,
    #                                     do_sample=True, eos_token_ids=999, num_return_sequences=1)
    outputs = model.generate(input_ids=input_ids, eos_token_id=1, pad_token_id=3, do_sample=True, num_return_sequences=1,
                                        max_length=text_size, min_length=25,
                                        top_p=top_p, top_k=top_k, temperature=temperature,
                                        repetition_penalty=1.2)
    generated_text = ''
    gen = vocab.to_tokens(outputs[0].squeeze().tolist())

    for tk in gen:
        generated_text += tk.replace('▁', ' ')
    sent = generated_text.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
    sent = sent.replace("<s>", "")
    sent = sent.replace("</s>", "")
    sent = auto_enter(sent)
    # print(sent)
    return sent



def main(args):
    wandb.init(project="bookathon")
    tok_path = get_tokenizer()
    model, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size
    epochs = args.n_epochs
    learning_rate = 3e-5
    wamup_steps = 2000
    max_seq_len = 1024

    print("Dataset Loading... ", end=" ")
    dataset = essayDataset("./data/processed_210119.pkl", vocab, tok)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("[[[Done]]]")

    model = model.to(device)
    wandb.watch(model)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=wamup_steps, num_training_steps=-1
    )
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    model.zero_grad()

    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(epochs):
        print(f"Epoch {epoch} started" + "=" * 30)
        model.train()
        for idx, ess in enumerate(data_loader):
            # """  max 시퀀스가 넘으면 슬라이싱 """
            mask = True
            do_train = True
            while mask:
                if len(ess) > max_seq_len:
                    work = ess[:max_seq_len]
                    ess = ess[max_seq_len:]
                elif len(ess) < 100:
                    mask = False
                    do_train = False
                else:
                    work = ess
                    mask = False
                if do_train:
                    ess_tensor = torch.tensor(work).unsqueeze(0).to(device)

                    outputs = model(ess_tensor, labels=ess_tensor)
                    loss, logits = outputs[:2]

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    
                    sum_loss = sum_loss + loss.detach().data
                    wandb.log({"Loss": loss.detach().data})

                    proc_seq_count = proc_seq_count + 1
                    if proc_seq_count == batch_size:
                        proc_seq_count = 0
                        batch_count += 1
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        model.zero_grad()

                    if batch_count == args.print_every:
                        print(f"average loss for 100 epoch {sum_loss // args.print_every}")
                        batch_count = 0
                        sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"gpt2_essay_{epoch}.pt"),
            )
            model.eval()
            output = generation_byt("일상적인 대화도 어렵게 되었다.", model, tok, vocab)
            with open(f'./outputs/results_{epoch}.txt', 'w') as f:
                f.write(output)
            print(f"epoch {epoch} results: {output}")


if __name__ == "__main__":
    args = define_argparser()
    main(args)
