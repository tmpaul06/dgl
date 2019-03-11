import numpy as np
import torch as th
import os
from dgl.data.utils import *
import spacy
from tqdm import tqdm

nlp = spacy.load('en')

_urls = {
    'wmt': 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/wmt14bpe_de_en.zip',
    'scripts': 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/transformer_scripts.zip',
}


def store_dependency_parses(in_filename, out_filename):
    """Create dependency parses in advance so that training is fast"""
    with open(in_filename, 'r') as f:
        input_lines = f.readlines()

        print('Preparing dependency tokens for {} sentences using {}'.format(len(input_lines), in_filename))
        # Batch write
        batch_size = min(max(len(input_lines) // 100, 100), 500)
        with open(out_filename, 'w') as out_f:
            for i in tqdm(range(0, len(input_lines), batch_size)):
                lines = input_lines[i:(i + batch_size + 1)]
                out_lines = list()
                for line in lines:
                    # Replace @ with ''. This is a cheap hack
                    line = line.replace('@', '').strip()
                    if not line:
                        continue
                    tokens = nlp(line)

                    line_deps = list()
                    for tok in tokens:
                        line_deps.append(str((tok.i, tok.head.i)).replace(' ', ''))
                    out_lines.append(' '.join(line_deps))
                out_f.write('\n'.join(out_lines))


def prepare_dataset(dataset_name):
    "download and generate datasets"
    script_dir = os.path.join('scripts')
    if not os.path.exists(script_dir):
        download(_urls['scripts'], path='scripts.zip')
        extract_archive('scripts.zip', 'scripts')

    directory = os.path.join('data', dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return
    if dataset_name == 'multi30k':
        os.system('bash scripts/prepare-multi30k.sh')
        # Pre-create dependency parses for train, valid and test
        for fi in ['train', 'val', 'test2016']:
            store_dependency_parses('data/multi30k/{}.en.atok'.format(fi), 'data/multi30k/{}_deps.en.atok'.format(fi))
    elif dataset_name == 'wmt14':
        download(_urls['wmt'], path='wmt14.zip')
        os.system('bash scripts/prepare-wmt14.sh')
    elif dataset_name == 'copy' or dataset_name == 'tiny_copy':
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
            open(os.path.join(directory, 'train.out'), 'w') as f_out:
            for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
            open(os.path.join(directory, 'valid.out'), 'w') as f_out:
            for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
            open(os.path.join(directory, 'test.out'), 'w') as f_out:
            for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
            for c in char_list:
                f.write(c + '\n')

    elif dataset_name == 'sort' or dataset_name == 'tiny_sort':
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
            open(os.path.join(directory, 'train.out'), 'w') as f_out:
            for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
            open(os.path.join(directory, 'valid.out'), 'w') as f_out:
            for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
            open(os.path.join(directory, 'test.out'), 'w') as f_out:
            for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
            for c in char_list:
                f.write(c + '\n')
