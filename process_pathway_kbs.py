import os

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB

paths = PathhierPaths()

path_kb_dirs = {
                "humancyc": paths.humancyc_raw_data_dir,
                "kegg": paths.kegg_raw_data_dir,
                "panther": paths.panther_raw_data_dir,
                "pid": paths.pid_raw_data_dir,
                "reactome": paths.reactome_raw_data_dir,
                "smpdb": paths.smpdb_raw_data_dir,
                "wikipathways": paths.wikipathways_raw_data_dir
                }

for kb_name, kb_dir in path_kb_dirs.items():
    print(kb_name)
    print(kb_dir)
    kb = PathKB(kb_name, kb_dir)
    kb.dump_pickle(os.path.join(paths.processed_data_dir, kb_name + '.pickle'))



