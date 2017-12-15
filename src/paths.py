import os

DEFAULT_BASE_DIR = "~/git/pathhier/"


class PathhierPaths:

    raw_data_folder = "raw_data"
    processed_data_folder = "processed_data"
    output_folder = "output"
    utils_folder = "utils"
    gedevo_folder = "gedevo"
    gedevo_exec = "gedevo"

    def __init__(self, base_dir=DEFAULT_BASE_DIR):
        self.base_dir = base_dir

    @property
    def raw_data_dir(self):
        return os.path.join(self.base_dir, self.raw_data_folder)

    @property
    def processed_data_dir(self):
        return os.path.join(self.base_dir, self.processed_data_folder)

    @property
    def output_dir(self):
        return os.path.join(self.base_dir, self.output_folder)

    @property
    def utils_dir(self):
        return os.path.join(self.base_dir, self.utils_folder)

    @property
    def gedevo_path(self):
        return os.path.join(self.utils_dir, self.gedevo_folder, self.gedevo_exec)

