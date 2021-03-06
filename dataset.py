from torch.utils.data import Dataset

from kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import pickle

def sentencePieceTokenizer():
    tok_path = get_tokenizer()
    sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

    return sentencepieceTokenizer


def koGPT2Vocab():
    cachedir = "~/kogpt2/"

    # download vocab
    vocab_info = tokenizer
    vocab_path = download(
        vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    )

    koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(
        vocab_path,
        mask_token=None,
        sep_token=None,
        cls_token=None,
        unknown_token="<unk>",
        padding_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return koGPT2_vocab


def toString(list):
    if not list:
        return ""
    result = ""

    for i in list:
        result = result + i
    return result


class essayDataset(Dataset):

    def __init__(self, file_path, vocab, tokenizer):
        self.file_path = file_path
        self.sentence_list = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        with open(file_path, 'rb') as f:
            essay = pickle.load(f)

        for script in essay:
            tokenized_script = tokenizer(str(script))
            if len(tokenized_script) > 100:
                index_of_words = (
                    [vocab[vocab.bos_token],]
                    + vocab[tokenized_script]
                    + [vocab[vocab.eos_token]]
                )
                self.sentence_list.append(index_of_words)
        print("sentence list length :", len(self.sentence_list))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, index):
        return self.sentence_list[index]
