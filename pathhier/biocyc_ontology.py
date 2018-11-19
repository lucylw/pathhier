from overrides import overrides

from pathhier.ontology import Ontology
import pathhier.utils.file_utils as file_utils


# class for representing biocyc ontology (has BP3 properties and types)
class BiocycOntology(Ontology):

    BIOCYC_CLASSES_ENDTAG = "//"

    def __init__(self,
                 name: str,
                 filename: str = None,
                 pathway_file: str = None):
        super().__init__(name, filename)
        self.pathway_file = pathway_file
        self.pw_classes = dict()

    @staticmethod
    def _chunkify(lines, delim):
        """
        Divide lines into chunks by delimiter
        :param lines: lines of text
        :param delim: delimiter patter
        :return:
        """
        chunk = []
        for line in lines:
            if not line:
                continue
            if line == delim:
                if len(chunk) > 0:
                    yield chunk
                    chunk = []
            elif line.startswith('/'):
                if len(line) > 1:
                    chunk[-1] += ' ' + line[1:]
            else:
                chunk.append(line)

        if len(chunk) > 0:
            yield chunk

    def _get_pathway_class_tree(self, classes):
        """
        From entity list, keep only pathway class hierarchy
        :param ents:
        :return:
        """

        def _form_dict(d):
            return {
                'name': d['names'][0] if d['names'] else '',
                'aliases': d['names'],
                'synonyms': d['synonyms'],
                'definition': d['comment'],
                'subClassOf': d['types'],
                'part_of': [],
                'instances': []
            }

        pathway_classes = dict()

        start_classes = {"Pathways"}
        matches = [cl for cl in classes if cl['uid'] in start_classes]
        for m in matches:
            pathway_classes[m['uid']] = _form_dict(m)

        while True:
            next_classes = [cl for cl in classes if set(cl['types']).intersection(start_classes)]
            for m in next_classes:
                pathway_classes[m['uid']] = _form_dict(m)
            start_classes = set([cl['uid'] for cl in next_classes])
            if not start_classes:
                break

        self.pw_classes = pathway_classes
        return

    @overrides
    def load_from_file(self):
        """
        Load both class and pathway dat files
        :return:
        """
        # all pathway classes
        classes = []

        for chunk in BiocycOntology._chunkify(
            file_utils.read_dat_lines(self.filename, comment='#'),
            BiocycOntology.BIOCYC_CLASSES_ENDTAG
        ):
            cls = dict()
            cls['types'] = []
            cls['names'] = []
            cls['synonyms'] = []
            cls['comment'] = []

            for line in chunk:
                if line.startswith('UNIQUE-ID - '):
                    # class id
                    cls['uid'] = line[len('UNIQUE-ID - '):].replace('-', ' ')
                elif line.startswith('TYPES - '):
                    # class type
                    cls['types'].append(line[len('TYPES - '):].replace('-', ' '))
                elif line.startswith('COMMENT - '):
                    # comment/definition
                    cls['comment'].append(line[len('COMMENT - '):])
                elif line.startswith('COMMON-NAME - '):
                    # common names
                    cls['names'].append(line[len('COMMON-NAME - '):])
                elif line.startswith('SYNONYMS - '):
                    # synonyms
                    cls['synonyms'].append(line[len('SYNONYMS - '):])

            classes.append(cls)

        self._get_pathway_class_tree(classes)

        # all pathway instances
        for chunk in BiocycOntology._chunkify(
            file_utils.read_dat_lines(self.pathway_file, comment='#'),
            BiocycOntology.BIOCYC_CLASSES_ENDTAG
        ):
            uid = None

            for line in chunk:
                if line.startswith('UNIQUE-ID - '):
                    # class id
                    uid = line[len('UNIQUE-ID - '):]
                elif line.startswith('TYPES - '):
                    # class type
                    typ = line[len('TYPES - '):].replace('-', ' ')
                    inst_class = {
                        'name': None,
                        'aliases': [],
                        'synonyms': [],
                        'definition': [],
                        'subClassOf': [typ],
                        'part_of': [],
                        'instances': []
                    }
                    self.pw_classes[uid] = inst_class
                    self.pw_classes[typ]['instances'].append(uid)
        return