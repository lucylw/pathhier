
import argparse

from pathhier.pw_aligner import PWAligner


def bootstrap(args):
    print('Bootstrapping mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.bootstrap_model(args.num_iter)


def train(args):
    print('Training mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.train_model(args.output_dir, args.batch_size, args.cuda_device)


def run(args):
    print('Alignment mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.run_model(
        args.name_model, args.def_model, args.output_dir, args.output_header, args.batch_size, args.cuda_device
    )


def bow(args):
    print('Bag of words mode')
    aligner = PWAligner(args.kb, args.pw)
    aligner.run_bow_model(
        args.output_dir, args.output_header
    )


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

bootstrap_parser = subparsers.add_parser('bootstrap')
bootstrap_parser.add_argument('kb')
bootstrap_parser.add_argument('pw')
bootstrap_parser.add_argument('num_iter', type=int)
bootstrap_parser.set_defaults(func=bootstrap)

train_parser = subparsers.add_parser('train')
train_parser.add_argument('kb')
train_parser.add_argument('pw')
train_parser.add_argument('batch_size', type=int)
train_parser.add_argument('cuda_device', type=int)
train_parser.add_argument('output_dir', type=str)
train_parser.set_defaults(func=train)

run_parser = subparsers.add_parser('run')
run_parser.add_argument('kb')
run_parser.add_argument('pw')
run_parser.add_argument('batch_size', type=int)
run_parser.add_argument('cuda_device', type=int)
run_parser.add_argument('name_model', type=str)
run_parser.add_argument('def_model', type=str)
run_parser.add_argument('output_dir', type=str)
run_parser.add_argument('output_header', type=str)
run_parser.set_defaults(func=run)

bow_parser = subparsers.add_parser('bow')
bow_parser.add_argument('kb')
bow_parser.add_argument('pw')
bow_parser.add_argument('output_dir', type=str)
bow_parser.add_argument('output_header', type=str)
bow_parser.set_defaults(func=bow)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)