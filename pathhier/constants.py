import numpy as np
from rdflib import Namespace

# Constants for pathhier project

# Biopax3 namespace
BP3 = Namespace('http://www.biopax.org/release/biopax-level3.owl#')

# Length for character ngrams
CHARACTER_NGRAM_LEN = 5

# IDF limit below which tokens are thrown out
IDF_LIMIT = np.log(20)

# number of top candidates to keep
KEEP_TOP_N_CANDIDATES = 20

# similarity score cutoff
SIMSCORE_THRESHOLD = 0.25

# number of bootstrap models
NUM_BOOTSTRAP_MODELS = 8

# bootstrap keep percentage
KEEP_TOP_N_PERCENT_MATCHES = 0.0025

# NN decision threshols
NN_DECISION_THRESHOLD = 0.5

# Development/Test set proportion
DEV_DATA_PORTION = 0.2
TEST_DATA_PORTION = 0.1

# Number of matches to keep per KB id
KEEP_TOP_N_MATCHES = 10

# weighting for name and definition
NAME_WEIGHT = 0.75
DEF_WEIGHT = 0.25
assert NAME_WEIGHT + DEF_WEIGHT == 1.

PATHWAY_KBS = ["humancyc",
               "kegg",
               "pid",
               "panther",
               "smpdb",
               "reactome",
               "wikipathways"]

PW_MAPPED_KBS = ["kegg",
                 "pid",
                 "smpdb"]

BIOPAX_RX_TYPES = ["TemplateReaction",
                   "TransportWithBiochemicalReaction",
                   "BiochemicalReaction",
                   "TemplateReactionRegulation",
                   "ComplexAssembly",
                   "MolecularInteraction",
                   "Transport"]

BIOPAX_NAME_PROPERTIES = ["displayName",
                          "standardName",
                          "name"]

GPML_ENTITY_TYPES = ["GeneProduct",
                     "Protein",
                     "Metabolite",
                     "SmallMolecule",
                     "Rna",
                     "Dna",
                     "Complex"]

ENT_TYPE_MAP = {'GeneProduct': 'Protein',
                'Metabolite': 'SmallMolecule'}

DB_XREF_MAP = {'cas': 'CAS',
               'chebi': 'ChEBI',
               'Chemspider': 'ChemSpider',
               'ENSEMBL': 'Ensembl',
               'Entrez Gene': 'Entrez',
               'gene ontology': 'GO',
               'KEGG Genes': 'KEGG',
               'KEGG Orthology': 'KEGG',
               'KEGG ortholog': 'KEGG',
               'KEGG-legacy': 'KEGG',
               'Kegg ortholog': 'KEGG',
               'kegg glycan': 'KEGG',
               'kegg pathway': 'KEGG',
               'KEGG Compound': 'KEGG',
               'miRBase mature sequence': 'miRBase',
               'miRBase Sequence': 'miRBase',
               'PubChem-compound': 'PubChem',
               'PubChem-substance': 'PubChem',
               'UniProt Isoform': 'UniProt',
               'Uniprot-SwissProt': 'UniProt',
               'Uniprot-TrEMBL': 'UniProt',
               'uniprot': 'UniProt',
               'uniprot knowledgebase': 'UniProt'}


KEEP_ENTITY_TYPES = ['Dna',
                     'SmallMolecule',
                     'Rna',
                     'Complex',
                     'Protein']

PATHWAY_XREF_AVOID_TERMS = ['pathwaycommons', 'biopax', 'pubmed']

ENTITY_XREF_AVOID_TERMS = ['pathwaycommons', 'biopax', 'pubmed', 'biocyc', 'BioCyc', 'Reactome']

KEEP_XREF_DBS = ['CAS',
                 'ChEBI',
                 'ChEMBL',
                 'ChemSpider',
                 'DrugBank',
                 'EMBL',
                 'Ensembl',
                 'EcoGene',
                 'Entrez',
                 'Enzyme Nomenclature',
                 'GeneOntology',
                 'GO',
                 'HGNC',
                 'HMDB',
                 'HumanCyc',
                 'KEGG',
                 'KNApSAcK',
                 'LIPID MAPS',
                 'ModBase',
                 'NCBI Nucleotide',
                 'NCBI Protein',
                 'NCI',
                 'PDB',
                 'Pfam',
                 'PubChem',
                 'RefSeq',
                 'Swiss-Model',
                 'TAIR',
                 'UMBBD-Compounds',
                 'UniGene',
                 'UniProt',
                 'Wikidata',
                 'Wikipedia',
                 'miRBase',
                 'pato',
                 'SMPDB']

BRIDGEDB_KEYS = {'KEGG': 'Kg',
                 'KEGG Compound': 'Ck',
                 'EcoGene': 'Ec',
                 'UniProt': 'S',
                 'NCBI Protein': 'Np',
                 'Ensembl': 'En',
                 'RefSeq': 'Q',
                 'PDB': 'Pd',
                 'UniGene': 'U',
                 'Swiss-Model': 'Sw',
                 'WikiData': 'Wd',
                 'Enzyme Nomenclature': 'E',
                 'GeneOntology': 'T',
                 'Entrez': 'L',
                 'HGNC': 'H',
                 'CAS': 'Ca',
                 'PubChem': 'Cpc',
                 'KNApSAcK': 'Cks',
                 'HMDB': 'Ch',
                 'ChemSpider': 'Cs',
                 'ChEBI': 'Ce',
                 'LIPID MAPS': 'Lm',
                 'EMBL': 'Em',
                 'miRBase': 'Mb'}

BRIDGEDB_MAP = {'UniProt': ['Ensembl', 'NCBI Protein', 'Entrez'],
                'NCBI Protein': ['UniProt', 'Ensembl', 'Entrez'],
                'Ensembl': ['UniProt', 'NCBI Protein', 'Entrez'],
                'Entrez': ['UniProt', 'Ensembl', 'NCBI Protein'],
                'ChEBI': ['PubChem', 'KEGG Compound', 'HMDB'],
                'PubChem': ['ChEBI', 'KEGG Compound', 'HMDB'],
                'KEGG Compound': ['ChEBI', 'PubChem', 'HMDB'],
                'HMDB': ['ChEBI', 'KEGG Compound', 'PubChem'],
                'EMBL': ['miRBase', 'Ensembl', 'Entrez'],
                'miRBase': ['EMBL', 'Ensembl', 'Entrez']}
