import os
import glob
import pickle
import itertools
from collections import defaultdict

from paths import PathhierPaths
from pathway_kb import PathwayKB


class PathwayKBLoader:
    def __init__(self):
        paths = PathhierPaths()
        self.kb_data_path = paths.raw_data_dir
        self.processed_path = paths.processed_data_dir
        self.kbs = dict()

    def get_kb_info(self):
        """
        Get the name and directory of KBs in the data directory
        :return: Dict {kb_name: kb_dir}
        """
        kb_info = dict()
        for subdir in glob.glob(self.kb_data_path + '*/'):
            name = os.path.basename(os.path.normpath(subdir))
            kb_info[name] = subdir
        return kb_info

    def load_pathway_kbs(self, kb_info):
        """
        Load all pathway raw data, process, and save processed data
        :param kb_info: dict with kb names and paths
        :return:
        """
        identifiers = []

        for kb_name, kb_path in kb_info.items():
            kb = PathwayKB(kb_name, kb_path)
            kb.load()
            identifiers += [ent.identifiers for ent in kb.entities]
            output_file = os.path.join(self.processed_path, 'kb-{}.json'.format(kb_name))
            kb.dump(output_file)

        identifier_file = os.path.join(self.processed_path, 'identifiers.pickle')
        pickle.dump(identifiers, open(identifier_file, 'wb'))
        return identifiers

    def build_identifier_dict(self, identifiers):
        """
        build an identifier dictionary of all identifier mappings
        :param identifiers:
        :return:
        """
        identifiers = self.merge_similar(identifiers)
        id_mapping_dict = defaultdict(set)
        for id_set in identifiers:
            for p, q in itertools.combinations(id_set, 2):
                id_mapping_dict[p].add(q)
                id_mapping_dict[q].add(p)
        id_mapping_dict = self.add_bridge_db_identifiers(id_mapping_dict)

if __name__ == "__main__":
    path_kb_loader = PathwayKBLoader()
    kb_paths = path_kb_loader.get_kb_info()
    ids = path_kb_loader.load_pathway_kbs(kb_paths)