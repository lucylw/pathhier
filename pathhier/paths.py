import os

DEFAULT_BASE_DIR = "/Users/lwang/git/pathhier/"


class PathhierPaths:

    data_folder = "data"
    raw_data_folder = "raw_data"
    processed_data_folder = "processed_data"
    other_data_folder = "other_data"
    pw_folder = "pathway_ontology"
    output_folder = "output"
    src_folder = "src"
    utils_folder = "utils"
    gedevo_folder = "gedevo"
    gedevo_exec = "gedevo"

    humancyc_dir = "humancyc"
    kegg_dir = "kegg"
    panther_dir = "panther"
    pid_dir = "pid"
    reactome_dir = "reactome"
    smpdb_dir = "smpdb"
    wikipath_dir = "wikipathways"

    def __init__(self, base_dir=DEFAULT_BASE_DIR):
        self.base_dir = base_dir

    @property
    def raw_data_dir(self):
        return os.path.join(self.base_dir, self.data_folder, self.raw_data_folder)

    @property
    def humancyc_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.humancyc_dir)

    @property
    def kegg_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.kegg_dir)

    @property
    def panther_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.panther_dir)

    @property
    def pid_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.pid_dir)

    @property
    def reactome_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.reactome_dir)

    @property
    def smpdb_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.smpdb_dir)

    @property
    def wikipathways_raw_data_dir(self):
        return os.path.join(self.raw_data_dir, self.wikipath_dir)

    @property
    def processed_data_dir(self):
        return os.path.join(self.base_dir, self.data_folder, self.processed_data_folder)

    @property
    def other_data_dir(self):
        return os.path.join(self.base_dir, self.data_folder, self.other_data_folder)

    @property
    def pathway_ontology_file(self):
        return os.path.join(self.base_dir, self.data_folder, self.pw_folder, "pw_20161021.xrdf")

    @property
    def output_dir(self):
        return os.path.join(self.base_dir, self.output_folder)

    @property
    def utils_dir(self):
        return os.path.join(self.base_dir, self.src_folder, self.utils_folder)

    @property
    def gedevo_path(self):
        return os.path.join(self.utils_dir, self.gedevo_folder, self.gedevo_exec)

