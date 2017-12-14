import os

from

DEFAULT_BASE_DIR = "~/git/pathhier/"


class PathhierPaths:

    data_folder = "data"
    output_folder = "output"
    utils_folder = "utils"
    gedevo_folder = "gedevo"
    gedevo_exec = "gedevo"

    def __init__(self, base_dir=DEFAULT_BASE_DIR):
        self.base_dir = base_dir

    @property
    def data_dir(self):
        return os.path.join(self.base_dir, self.data_folder)

    @property
    def output_dir(self):
        return os.path.join(self.base_dir, self.output_folder)

    @property
    def utils_dir(self):
        return os.path.join(self.base_dir, self.utils_folder)

    @property
    def gedevo_path(self):
        return os.path.join(self.utils_dir, self.gedevo_folder, self.gedevo_exec)
