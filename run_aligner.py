
import argparse

from pathhier.pw_aligner import PWAligner


def train(args):
    print('Training mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.train_model(args.num_iter, args.batch_size, args.cuda_device)


def run(args):
    print('Alignment mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.run_model(
        args.name_model, args.def_model, args.batch_size, args.cuda_device
    )


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train')
train_parser.add_argument('kb')
train_parser.add_argument('pw')
train_parser.add_argument('batch_size', type=int)
train_parser.add_argument('cuda_device', type=int)
train_parser.add_argument('num_iter', type=int)
train_parser.set_defaults(func=train)

run_parser = subparsers.add_parser('run')
run_parser.add_argument('name_model')
run_parser.add_argument('def_model')
run_parser.add_argument('kb')
run_parser.add_argument('pw')
run_parser.add_argument('batch_size', type=int)
run_parser.add_argument('cuda_device', type=int)
run_parser.set_defaults(func=run)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)