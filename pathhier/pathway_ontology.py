from overrides import overrides
from typing import List

from rdflib import Namespace
from rdflib.namespace import RDF, RDFS

from pathhier.ontology import Ontology


# class for representing pathway ontology
class PathwayOntology(Ontology):
    def __init__(self,
                 name: str,
                 filename: str = None):
        super().__init__(name, filename)

    @overrides
    def get_synonyms(self, uri) -> (List, List):
        """
        Get synonyms of object given by uri
        :param uri:
        :return:
        """
        synonyms = []
        annotations = []
        for syn in self.graph.objects(uri, self.oboInOwl_hasExactSynonym):
            synonyms.append(syn.value)
        for syn in self.graph.objects(uri, self.oboInOwl_hasRelatedSynonym):
            if ':' in syn.value:
                annotations.append(syn.value)
            else:
                synonyms.append(syn.value)
        return synonyms, annotations