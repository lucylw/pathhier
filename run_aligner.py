
import argparse

from pathhier.pw_aligner import PWAligner


def bootstrap(args):
    print('Bootstrapping mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.bootstrap_model(args.num_iter, args.batch_size, args.cuda_device)


def train(args):
    print('Training mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.train_model(args.output_dir, args.batch_size, args.cuda_device)


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
train_parser.add_argument('output_dir', type=str)
train_parser.set_defaults(func=train)

bootstrap_parser = subparsers.add_parser('bootstrap')
bootstrap_parser.add_argument('kb')
bootstrap_parser.add_argument('pw')
bootstrap_parser.add_argument('batch_size', type=int)
bootstrap_parser.add_argument('cuda_device', type=int)
bootstrap_parser.add_argument('num_iter', type=int)
bootstrap_parser.set_defaults(func=bootstrap)

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