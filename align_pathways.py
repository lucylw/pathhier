import argparse

from pathhier.pathway_aligner import PathAligner


def align(args):
    print('Aligning pathways')
    aligner = PathAligner(
        args.pathway_pairs,
        s2v_path=args.s2v_path,
        w2v_file=args.w2v_file,
        ft_file=args.ft_file,
        start_ind=args.start_ind,
        end_ind=args.end_ind
    )
    aligner.align_pathways()


def enrich(args):
    print('Enriching pathways')
    aligner = PathAligner(
        args.pathway_pairs,
        s2v_path=args.s2v_path
    )
    aligner.enrich_only()


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

align_parser = subparsers.add_parser('align')
align_parser.add_argument('--pathway_pairs', dest='pathway_pairs')
align_parser.add_argument('--s2v_path', dest='s2v_path')
align_parser.add_argument('--w2v_file', dest='w2v_file')
align_parser.add_argument('--ft_file', dest='ft_file')
align_parser.add_argument('--start_ind', dest='start_ind', type=int)
align_parser.add_argument('--end_ind', dest='end_ind', type=int)
align_parser.set_defaults(func=align)

enrich_parser = subparsers.add_parser('enrich')
enrich_parser.add_argument('--pathway_pairs', dest='pathway_pairs')
enrich_parser.add_argument('--s2v_path', dest='s2v_path')
enrich_parser.set_defaults(func=enrich)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)