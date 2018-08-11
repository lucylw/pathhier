#!python
# script for preprocessing all pathway data (kbs, identifiers, pathway ontology)

from pathhier.pathway_kb_loader import PathwayKBLoader


path_kb_loader = PathwayKBLoader()

# # process all raw kbs
# path_kb_loader.process_raw_pathway_kbs()
#
# # load processed kbs
# path_kb_loader.load_raw_pathway_kbs()
#
# # extract identifiers
# path_kb_loader.get_identifier_map()
#
# # merge entities with shared identifiers
# path_kb_loader.merge_entities_on_identifiers()

# process pathway ontology and save
path_kb_loader.process_pathway_ontology()
