
import torch
import torch.nn.functional as F
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp
import logging
import os

pytorch_kogpt2 = {
    'url':
        './checkpoint/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits

def auto_enter(text):
    text = (text.replace("   ", "\n"))
    text = text.split("\n")

    text = [t.lstrip() for t in text if t != '']
    return "\n\n".join(text)

class GPT2:
    def __init__(self, load_path):
        ctx = 'cuda'
        cachedir = '~/kogpt2/'
        org_path = "trained_models/gpt2_essay_15.pt"

        # download vocab
        vocab_info = tokenizer
        vocab_path = download(vocab_info['url'],
                              vocab_info['fname'],
                              vocab_info['chksum'],
                              cachedir=cachedir)
        # Device ??????
        self.device = torch.device(ctx)

        # ????????? Checkpoint ????????????

        checkpoint = torch.load(load_path, map_location=self.device)
        # 1013: special token ????????? ?????? keys ?????? ???????????? ?????? ?????? ?????? ??????
        checkpoint_org = torch.load(org_path, map_location=self.device)
        ckpt_final = {k:v for k, v in zip(checkpoint_org.keys(), checkpoint.values())} # ?????? state_dict ??? value ??? ????????? ?????? ????????? ?????????

        # KoGPT-2 ?????? ?????? ????????? ?????? GPT2LMHeadModel ??????
        self.kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        self.kogpt2model.load_state_dict(ckpt_final)
        self.kogpt2model.to(self.device)


        self.kogpt2model.eval()
        self.vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                  mask_token=None,
                                                                  sep_token=None,
                                                                  cls_token=None,
                                                                  unknown_token='<unk>',
                                                                  padding_token='<pad>',
                                                                  bos_token='<s>',
                                                                  eos_token='</s>')

        tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer(tok_path)

    def generation(self, input_sentence, temperature=0.85, top_p=0.9, top_k=50, text_size=100):
        sent = ''
        sent = sent + input_sentence
        sent = self.sample_sequence(self.kogpt2model, self.tok, self.vocab, sent, text_size, temperature, top_p, top_k)
        sent = sent.replace("//", "\n")  # ????????????????????? ????????? ????????? ??????
        sent = sent.replace("</s>", "")
        sent = auto_enter(sent)
        return sent

    def sample_sequence(self, model, tok, vocab, sent, text_size, temperature, top_p, top_k):
        toked = tok(sent)  # ?????? ??????
        count = 0
        generated_text = ''

        if len(toked) > 1024:
            return 0

        while 1:  # ???????????? ???????????? ?????????.
            # ?????? ?????? ??????
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
            input_ids = input_ids.to(self.device)

            predicts = model(input_ids)
            pred = predicts[0]

            # temperature ??????
            logits = pred
            logits = logits[:, -1, :] / temperature
            # top k
            logits = top_k_logits(logits, top_k)
            # top p
            logits = top_p_logits(logits, top_p=top_p)

            logits = logits.to(self.device)

            # ???????????? ??????
            log_probs = F.softmax(logits, dim=-1)
            # ?????? ?????? ???????????? ?????? ????????? ??????
            prev = torch.multinomial(log_probs, num_samples=1)
            # ?????? ????????? (???????????? gpt2??? ?????? ??????)
            gen = vocab.to_tokens(prev.squeeze().tolist())

            # ????????? ??????????????? ????????? ??????.
            if gen == '</s>' or count > text_size:
                sent += gen.replace('???', ' ')
                generated_text += gen.replace('???', ' ')
                sent += '\n'
                generated_text += '\n'

                break

            sent += gen.replace('???', ' ')
            generated_text += gen.replace('???', ' ')
            toked = tok(sent)

        return sent

    def generation_byt(self, input_sentence, temperature=0.8, top_p=0.95, top_k=50, text_size=200):
        ctx = 'cuda'
        device = torch.device(ctx)

        sent = ''
        sent = sent + input_sentence
        toked = self.tok(sent)

        input_ids = torch.tensor([self.vocab[self.vocab.bos_token], ] +  self.vocab[toked]).unsqueeze(0)
        input_ids = input_ids.to(ctx)
        # outputs = self.kogpt2model.generate(input_ids=input_ids, max_length=200, repetition_penalty=1.2, pad_token_id=3,
        #                                     do_sample=True, eos_token_ids=999, num_return_sequences=1)
        outputs = self.kogpt2model.generate(input_ids=input_ids, eos_token_id=1, pad_token_id=3, do_sample=True, num_return_sequences=1,
                                            max_length=text_size, min_length=25,
                                            top_p=top_p, top_k=top_k, temperature=temperature,
                                            repetition_penalty=1.3)
        generated_text = ''
        gen = self.vocab.to_tokens(outputs[0].squeeze().tolist())

        for tk in gen:
            generated_text += tk.replace('???', ' ')
        sent = generated_text.replace("//", "\n")  # ????????????????????? ????????? ????????? ??????
        # sent = sent.replace(input_sentence, "")
        sent = sent.replace("<s>", "")
        sent = sent.replace("</s>", "")
        sent = auto_enter(sent)
        return sent

if __name__ == "__main__":
    # logging setup
    log_dir = './logs'
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    from datetime import datetime
    cur = datetime.now().strftime(r"%m%d_%H%M")

    logger = logging.getLogger(__name__)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, f"./gpt2_{cur}.log"))
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.setLevel(level=logging.DEBUG)
    
    model = GPT2("trained_models/gpt2_essay_15.pt") # ????????? ????????? ???????????? ????????? ???????????? 100 epoch???
    sizes = [100]
    ex1 = "????????? ?????? ????????? ?????? ???????????? ????????? ????????? ?????? ????????? ????????? ??????????????? ?????????"
    temp = 0.8
    topp = 0.95
    topk = 40
    logger.debug(f"params:  temp: {temp}  top_p: {topp}  top_k: {topk}")
    logger.debug(f"input: {ex1}")
    for size in sizes:
        logger.debug(f"\nsize: {size}")
        for i in range(5):
            logger.debug(f"{i}-th output: {model.generation_byt(input_sentence=ex1, temperature=temp, top_p=topp, top_k=topk, text_size=size)}")