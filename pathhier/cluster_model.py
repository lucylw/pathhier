import os
import sys
import csv
import json
from collections import defaultdict

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB, Pathway, Group
from pathhier.extract_training_data import TrainingDataExtractor
import pathhier.constants as constants
import pathhier.utils.pathway_utils as pathway_utils


# class for clustering pathways based on the output of the PW alignment algorithm
class PathwayClusterer:
    def __init__(self):
        """
        Initialize class
        """
        paths = PathhierPaths()
        self.output_dir = os.path.join(paths.output_dir, 'model_output')

        # load PW
        pw_file_path = os.path.join(paths.pathway_ontology_dir, 'pw.json')
        self.pw = self._load_pw_from_file(pw_file_path)

        # load KBs
        self.kbs = dict()
        for kb_name in paths.all_kb_paths:
            kb_file_path = os.path.join(paths.processed_data_dir, 'kb_{}.pickle'.format(kb_name))
            self.kbs[kb_name] = self._load_kb_from_file(kb_name, kb_file_path)

        # load PID mapping file
        pid_mapping_file = os.path.join(paths.other_data_dir, 'pid_mapping_dict.json')
        self.pid_mapping_dict = self._load_pid_mapping_file(pid_mapping_file)

    @staticmethod
    def _load_pw_from_file(pw_file):
        """
        Load PW from json file
        :param pw_file:
        :return:
        """
        print("Loading pathway ontology...")
        assert os.path.exists(pw_file)
        with open(pw_file, 'r') as f:
            pw = json.load(f)
        return pw

    @staticmethod
    def _load_kb_from_file(kb_name, kb_file):
        """
        Load KB from pickle file
        :param kb_name:
        :param kb_file:
        :return:
        """
        print("Loading {}...".format(kb_name))
        assert (os.path.exists(kb_file))
        kb = PathKB(kb_name)
        kb = kb.load_pickle(kb_name, kb_file)
        return kb

    @staticmethod
    def _load_pid_mapping_file(pid_map_file):
        """
        Load PID mapping file
        :param pid_map_file:
        :return:
        """
        print('Loading PID mapping dict from file...')
        assert (os.path.exists(pid_map_file))
        with open(pid_map_file, 'r') as f:
            pid_mapping_dict = json.load(f)
        return pid_mapping_dict

    @staticmethod
    def _at_least_two(entry):
        """
        Check if dict entry has pathways from at least two KBs
        :return:
        """
        kb_exist = [k for k, v in entry.items() if len(v) > 0]
        return len(kb_exist) > 1

    @staticmethod
    def _get_all_ent_xrefs(pathway: Pathway):
        """
        Return list of all entities (physical) and xrefs
        :param pathway:
        :return:
        """
        keep_types = ['SmallMolecule', 'Protein', 'Dna', 'Rna']

        xref_sets = []
        for ent in pathway.entities:
            xrefs = []
            if type(ent) == Group:
                continue
            if ent.obj_type in keep_types:
                if ent.obj_type == 'Complex':
                    xrefs = pathway.get_all_complex_xrefs(ent)
                else:
                    xrefs = ent.xrefs
                xrefs = pathway_utils.clean_xrefs(xrefs, constants.ENTITY_XREF_AVOID_TERMS)
            if xrefs:
                xref_sets.append(tuple(set(xrefs)))

        return set(xref_sets)

    # OBSOLETE
    def get_pid_mappings(self, pid):
        """
        Get PID mappings between PID uids and PW mappings
        :return:
        """
        extractor = TrainingDataExtractor()
        pid_names = {k: v for k, v in extractor.kb_path_names.items() if k.startswith('PID')}

        mapping_dict = dict()
        for pid_uid, pid_data in pid_names.items():
            pid_name, _ = pid_data
            pid_name_lower = pid_name.lower()

            matching_paths = [p for p in pid if p.name.lower() == pid_name_lower]
            if len(matching_paths) > 1:
                print('More than 1 match!')
                print(matching_paths)
            for match in matching_paths:
                mapping_dict[pid_uid] = match.uid

        return mapping_dict

    def get_positive_mappings_from_pw(self):
        """
        Get all positive mappings from PW
        :return:
        """
        print('Getting mappings from PW...')
        pw_match_dict = defaultdict(list)

        for pw_id, pw_value in self.pw.items():
            xrefs = pw_value['synonyms']

            for xref in xrefs:
                kb_name = None
                kb_id = None

                xref_db, xref_id = xref.split(':')

                if 'KEGG' in xref_db:
                    kb_name = 'kegg'
                    kb_id = 'hsa' + xref_id
                elif 'SMP' in xref_db:
                    kb_name = 'smpdb'
                    kb_id = xref_id
                elif 'PID' in xref_db:
                    kb_name = 'pid'
                    kb_id = xref

                if kb_name and kb_id:
                    pw_match_dict[pw_id].append(
                        (1., kb_name, kb_id)
                    )

        return pw_match_dict

    def load_all_pw_alignment_outputs(self, pw_dict):
        """
        Iterate through newly mapped KBs and read matches from output files
        :param pw_dict:
        :return:
        """
        for kb_name in constants.PATHWAY_KBS:
            if kb_name not in constants.PW_MAPPED_KBS:
                output_file = os.path.join(self.output_dir, '{}_final_matches.tsv'.format(kb_name))
                if os.path.exists(output_file):
                    print('Loading alignment results for {}...'.format(kb_name))
                    with open(output_file, 'r') as f:
                        reader = csv.reader(f, delimiter='\t')
                        next(reader)
                        for score, path_id, _, _, pw_id, _, _ in reader:
                            if ':' in path_id:
                                _, kb_id = path_id.split(':')
                            else:
                                kb_id = path_id
                            pw_dict[pw_id].append(
                                (float(score), kb_name, kb_id)
                            )
                else:
                    print("Skipping: file doesn't exist... {}".format(output_file))

        return pw_dict

    def group_by_kb(self, pw_dict):
        """
        Group all PW alignment outputs by KB
        :param pw_dict:
        :return:
        """
        print('Grouping by KB...')
        group_by_kb = dict()

        for pw_id, cluster in pw_dict.items():

            cluster_dict = dict()

            for kb_name in constants.PATHWAY_KBS:
                matches = [(score, uid) for score, name, uid in cluster if name == kb_name]
                matches.sort(key=lambda x: x[0], reverse=True)

                path_ents = []

                if len(matches) > 10:
                    matches = matches[:10]
                for m in matches:
                    uid = m[1]
                    if kb_name == 'humancyc':
                        uid = '{}:{}'.format('HumanCyc', m[1])
                    elif kb_name == 'kegg':
                        uid = '{}:{}'.format(kb_name, m[1])
                    elif kb_name == 'panther':
                        uid = '{}:{}'.format(kb_name, m[1])
                    elif kb_name == 'pid':
                        if m[1] in self.pid_mapping_dict:
                            uid = self.pid_mapping_dict[m[1]]
                        else:
                            uid = m[1]
                    elif kb_name == 'reactome':
                        uid = '{}:{}'.format('Reactome', m[1])
                    elif kb_name == 'smpdb':
                        uid = 'SMP{}'.format(m[1])

                    pathway = self.kbs[kb_name].get_pathway_by_uid(uid)

                    if pathway:
                        ent_xrefs = self._get_all_ent_xrefs(pathway)
                        path_ents.append([m[0], uid, ent_xrefs])
                    else:
                        print("Can't find {}".format(uid))
                        path_ents.append([m[0], uid, set([])])

                cluster_dict[kb_name] = path_ents

            group_by_kb[pw_id] = cluster_dict

        return group_by_kb

    def combine_entities(self, pw_dict):
        """
        Construct an entity mapping dict and merge entities
        :param pw_dict:
        :return:
        """
        ind = 0

        forward_dict = defaultdict(set)
        backward_dict = dict()

        empty_skip_count = 0
        group_skip_count = 0

        for pw_id, v in pw_dict.items():
            for kb_name, pathways in v.items():
                for score, kb_id, ent_xrefs in pathways:
                    for ent in ent_xrefs:
                        if len(ent) == 0:
                            empty_skip_count += 1
                            continue
                        if len(ent) > 10:
                            group_skip_count += 1
                            continue
                        exists = list(set([backward_dict.get(x) for x in ent if x in backward_dict]))

                        if len(exists) >= 1:
                            use_id = exists[0]

                            for merge_id in exists[1:]:
                                forward_dict[use_id].update(forward_dict[merge_id])
                                for x in forward_dict[merge_id]:
                                    backward_dict[x] = use_id
                                del forward_dict[merge_id]

                            forward_dict[use_id].update(set(ent))
                            for x in ent:
                                if x not in backward_dict:
                                    backward_dict[x] = use_id
                                else:
                                    assert backward_dict[x] == use_id
                        else:
                            use_id = ind
                            forward_dict[use_id].update(set(ent))
                            for x in ent:
                                assert x not in backward_dict
                                backward_dict[x] = use_id
                            ind += 1

        print('Skipped {} entities for having NO xrefs...'.format(empty_skip_count))
        print('Skipped {} entities for having > 10 xrefs, assume Group...'.format(group_skip_count))

        # convert mapping dict to use entitiy numbers
        mapped_pw_dict = dict()

        for pw_id, v in pw_dict.items():
            mapped_pw_dict[pw_id] = dict()

            for kb_name, pathways in v.items():
                mapped_pw_dict[pw_id][kb_name] = []
                for score, kb_id, ent_xrefs in pathways:
                    entities = []
                    for ent in ent_xrefs:
                        if len(ent) == 0 or len(ent) > 10:
                            continue
                        else:
                            matches = [backward_dict.get(x) for x in ent]
                            assert all([m == matches[0] for m in matches])
                            entities.append(matches[0])
                    mapped_pw_dict[pw_id][kb_name].append(
                        (score, kb_id, set(entities))
                    )
        return forward_dict, backward_dict, mapped_pw_dict

    def process_all(self):
        """
        Process all output data
        :return:
        """
        pw_mapping_dict = self.get_positive_mappings_from_pw()
        pw_mapping_dict = self.load_all_pw_alignment_outputs(pw_mapping_dict)
        pw_to_kb = self.group_by_kb(pw_mapping_dict)
        pw_to_kb = {k: v for k, v in pw_to_kb.items() if self._at_least_two(v)}
        return pw_to_kb


if __name__ == '__main__':
    clusterer = PathwayClusterer()
    pw_to_kb_dict = clusterer.process_all()
    ent_to_xrefs, xrefs_to_ent, pathway_ent_dict = clusterer.combine_entities(pw_to_kb_dict)
    