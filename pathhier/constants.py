# Constants for pathhier project

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
               'KEGG Compound': 'KEGG',
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
