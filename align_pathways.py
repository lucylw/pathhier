import argparse

from pathhier.pathway_aligner import PathAligner


def align(args):
    print('Aligning pathways')
    aligner = PathAligner(
        args.pathway_pairs,
        s2v_path=args.s2v_path,
        w2v_file=args.w2v_file,
        ft_file=args.ft_file
    )
    aligner.align_pathways(args.pairs_to_align, args.temp_folder, args.out_folder)


def enrich(args):
    print('Enriching pathways')
    aligner = PathAligner(
        args.pathway_pairs,
        s2v_path=args.s2v_path
    )
    aligner.enrich_only(args.pairs_to_align, args.temp_folder, args.out_folder)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

align_parser = subparsers.add_parser('align')
align_parser.add_argument('--pathway_pairs', dest='pathway_pairs')
align_parser.add_argument('--s2v_path', dest='s2v_path')
align_parser.add_argument('--w2v_file', dest='w2v_file')
align_parser.add_argument('--ft_file', dest='ft_file')
align_parser.add_argument('--pairs_to_align', dest='pairs_to_align')
align_parser.add_argument('--temp_folder', dest='temp_folder')
align_parser.add_argument('--out_folder', dest='out_folder')
align_parser.set_defaults(func=align)

enrich_parser = subparsers.add_parser('enrich')
enrich_parser.add_argument('--pathway_pairs', dest='pathway_pairs')
enrich_parser.add_argument('--s2v_path', dest='s2v_path')
enrich_parser.add_argument('--pairs_to_align', dest='pairs_to_align')
enrich_parser.add_argument('--temp_folder', dest='temp_folder')
enrich_parser.add_argument('--out_folder', dest='out_folder')
enrich_parser.set_defaults(func=enrich)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)