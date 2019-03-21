# Beam Search Module

from modules import *
from dataset import *
from tqdm import tqdm
import numpy as n
import argparse

k = 5 # Beam size

if __name__ == '__main__':

    class TransformerArgs:

        def __init__(self):
            self.dataset = 'multi30k'
            self.devices = ['cpu']
            self.ngpu = 1
            self.N = 2
            self.use_deps = True
            self.batch = 64
            self.viz = True
            self.universal = False
            self.print = False
            self.grad_accum = 1
            self.epoch = 20
            self.num_heads = 2
            self.gpus = '-1'

        def get_exp_setting(self):
            """Get a unique setting for the model params"""
            args_filter = ['batch', 'gpus', 'viz', 'master_ip', 'master_port', 'grad_accum', 'ngpu']
            sorted_keys = sorted(vars(args).keys())
            exp_arr = list()
            for k in sorted_keys:
                if k not in args_filter:
                    v = vars(args)[k]
                    exp_arr.append('{}'.format(v if not isinstance(v, list) else '.'.join(v)))
            return '-'.join(exp_arr)

    args = TransformerArgs()

    exp_setting = args.get_exp_setting()
    device = args.devices[0]

    dataset = get_dataset(args.dataset)
    V = dataset.vocab_size
    dim_model = 128

    fpred = open('pred.txt', 'w')
    fref = open('ref.txt', 'w')

    graph_pool = GraphPool()
    model = make_model(V, V, N=args.N, dim_model=dim_model, h=args.num_heads, device=device)
    with open('checkpoints/{}.pkl'.format(exp_setting), 'rb') as f:
        model.load_state_dict(th.load(f, map_location=lambda storage, loc: storage))
    model = model.to(device)
    model.eval()
    test_iter = dataset(graph_pool, mode='test', batch_size=args.batch, devices=[device], k=k, run_args=args)
    for i, g in enumerate(test_iter):
        with th.no_grad():
            output = model.infer(g, dataset.MAX_LENGTH, dataset.eos_id, k, alpha=0.6)
        for line in dataset.get_sequence(output):
            if args.print:
                print(line)
            print(line, file=fpred)
        for line in dataset.tgt['test']:
            print(line.strip(), file=fref)
    fpred.close()
    fref.close()
    os.system('bash chmod +x scripts/bleu.sh')
    os.system(r'bash scripts/bleu.sh pred.txt ref.txt')
    os.remove('pred.txt')
    os.remove('ref.txt')

