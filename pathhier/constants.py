import numpy as np

# Constants for pathhier project

# Length for character ngrams
CHARACTER_NGRAM_LEN = 5

# IDF limit below which tokens are thrown out
IDF_LIMIT = np.log(20)

# number of top candidates to keep
KEEP_TOP_N_CANDIDATES = 20

PATHWAY_KBS = ["humancyc",
               "kegg",
               "pid",
               "panther",
               "smpdb",
               "reactome",
               "wikipathways"]

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

DB_XREF_MAP = {'chebi': 'ChEBI',
               'Chemspider': 'ChemSpider',
               'ENSEMBL': 'Ensembl',
               'Entrez Gene': 'Entrez',
               'KEGG Genes': 'KEGG',
               'KEGG Orthology': 'KEGG',
               'KEGG ortholog': 'KEGG',
               'KEGG-legacy': 'KEGG',
               'Kegg ortholog': 'KEGG',
               'miRBase mature sequence': 'miRBase',
               'miRBase Sequence': 'miRBase',
               'PubChem-compound': 'PubChem',
               'PubChem-substance': 'PubChem',
               'UniProt Isoform': 'UniProt',
               'Uniprot-SwissProt': 'UniProt',
               'Uniprot-TrEMBL': 'UniProt',
               'uniprot': 'UniProt'}


KEEP_ENTITY_TYPES = ['Dna',
                     'SmallMolecule',
                     'Rna',
                     'Complex',
                     'Protein']

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
                 'HGNC',
                 'HMDB',
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
                 'pato']

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
