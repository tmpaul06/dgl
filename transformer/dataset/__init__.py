from .graph import *
from .fields import *
from .utils import prepare_dataset
import time
import os
import re
import random


class ClassificationDataset(object):
    "Dataset class for classification task."
    def __init__(self):
        raise NotImplementedError


class TranslationDataset(object):
    '''
    Dataset class for translation task.
    By default, the source language shares the same vocabulary with the target language.
    '''
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    MAX_LENGTH = 50
    def __init__(self, path, exts, train='train', valid='valid', test='test', vocab='vocab.txt', replace_oov=None):
        vocab_path = os.path.join(path, vocab)
        max_len = None
        self.src = {}
        self.tgt = {}
        with open(os.path.join(path, train + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['train'] = f.readlines()[:max_len]
        with open(os.path.join(path, train + '_deps.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['train-deps'] = f.readlines()[:max_len]
        with open(os.path.join(path, train + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['train'] = f.readlines()[:max_len]
        with open(os.path.join(path, valid + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['valid'] = f.readlines()[:max_len]
        with open(os.path.join(path, valid + '_deps.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['valid-deps'] = f.readlines()[:max_len]
        with open(os.path.join(path, valid + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['valid'] = f.readlines()[:max_len]
        with open(os.path.join(path, test + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['test'] = f.readlines()[:max_len]
        with open(os.path.join(path, test + '_deps.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['test-deps'] = f.readlines()[:max_len]
        with open(os.path.join(path, test + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['test'] = f.readlines()[:max_len]

        if not os.path.exists(vocab_path):
            self._make_vocab(vocab_path)

        vocab = Vocab(init_token=self.INIT_TOKEN,
                      eos_token=self.EOS_TOKEN,
                      pad_token=self.PAD_TOKEN,
                      unk_token=replace_oov)
        vocab.load(vocab_path)
        self.vocab = vocab
        strip_func = lambda x: x[:self.MAX_LENGTH]
        self.src_field = Field(vocab,
                               preprocessing=None,
                               postprocessing=strip_func)
        self.tgt_field = Field(vocab,
                               preprocessing=lambda seq: [self.INIT_TOKEN] + seq + [self.EOS_TOKEN],
                               postprocessing=strip_func)

    def get_seq_by_id(self, idx, mode='train', field='src'):
        "get raw sequence in dataset by specifying index, mode(train/valid/test), field(src/tgt)"
        if field == 'src':
            return self.src[mode][idx].strip().split()
        else:
            return [self.INIT_TOKEN] + self.tgt[mode][idx].strip().split() + [self.EOS_TOKEN]

    def _make_vocab(self, path, thres=2):
        word_dict = {}
        for mode in ['train', 'valid', 'test']:
            for line in self.src[mode] + self.tgt[mode]:
                for token in line.strip().split():
                    if token not in word_dict:
                        word_dict[token] = 0
                    else:
                        word_dict[token] += 1

        with open(path, 'w') as f:
            for k, v in word_dict.items():
                if v > 2:
                    print(k, file=f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self):
        return self.vocab[self.INIT_TOKEN]

    @property
    def eos_id(self):
        return self.vocab[self.EOS_TOKEN]

    def __call__(self, graph_pool, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1, run_args=None):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            k: beam size(only required for test)
            device: str or torch.device
            dev_rank: rank (id) of current device
            ndev: number of devices
            run_args: The runtime args
        '''
        src_data, tgt_data = self.src[mode], self.tgt[mode]
        n = len(src_data)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: partition then shuffle may not be equivalent to shuffle then
        # partition
        order = list(range(dev_rank, n, ndev))
        if mode == 'train':
            random.shuffle(order)

        src_buf, tgt_buf = [], []
        src_deps = []

        for idx in order:

            src_sample = self.src_field(
                src_data[idx].strip().split())
            tgt_sample = self.tgt_field(
                tgt_data[idx].strip().split())
            if not len(src_sample) or not len(tgt_sample):
                continue
            src_buf.append(src_sample)
            tgt_buf.append(tgt_sample)
            if run_args.use_deps:
                src_deps.append([tuple(map(int, re.findall(r'[0-9]+', el))) for el in self.src['{}-deps'.format(mode)][idx].split(' ')])
            else:
                src_deps.append(None)
            if len(src_buf) == batch_size:
                if mode == 'test':
                    yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, run_args.N, run_args.num_heads, device=device, src_deps=src_deps, vocab=self.vocab)
                else:
                    yield graph_pool(src_buf, tgt_buf, run_args.N, run_args.num_heads, device=device, src_deps=src_deps, vocab=self.vocab)
                src_buf, tgt_buf = [], []
                src_deps = []

        if len(src_buf) != 0:
            if mode == 'test':
                yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, run_args.N, run_args.num_heads, device=device, src_deps=src_deps, vocab=self.vocab)
            else:
                yield graph_pool(src_buf, tgt_buf, run_args.N, run_args.num_heads, device=device, src_deps=src_deps, vocab=self.vocab)

    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        ret = []
        filter_list = set([self.pad_id, self.sos_id, self.eos_id])
        for seq in batch:
            try:
                l = seq.index(self.eos_id)
            except:
                l = len(seq)
            ret.append(' '.join(self.vocab[token] for token in seq[:l] if not token in filter_list))
        return ret

def get_dataset(dataset):
    "we wrapped a set of datasets as example"
    prepare_dataset(dataset)
    if dataset == 'babi':
        raise NotImplementedError
    elif dataset == 'copy' or dataset == 'sort':
        return TranslationDataset(
            'data/{}'.format(dataset),
            ('in', 'out'),
            train='train',
            valid='valid',
            test='test',
        )
    elif dataset == 'multi30k':
        return TranslationDataset(
            'data/multi30k',
            ('en.atok', 'de.atok'),
            train='train',
            valid='val',
            test='test2016',
            replace_oov='<unk>'
        )
    elif dataset == 'wmt14':
        return TranslationDataset(
            'data/wmt14',
            ('en', 'de'),
            train='train.tok.clean.bpe.32000',
            valid='newstest2013.tok.bpe.32000',
            test='newstest2014.tok.bpe.32000.ende',
            vocab='vocab.bpe.32000')
    else:
        raise KeyError()
