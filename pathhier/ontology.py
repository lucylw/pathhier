import os

from typing import List

from rdflib import Graph
from rdflib.term import URIRef
from rdflib.namespace import RDF, RDFS, OWL

from pathhier.paths import PathhierPaths


# thin wrapper around rdf graph class
class Ontology:

    # URIs in Pathway Ontology
    oboInOwl_hasExactSynonym = URIRef('http://www.geneontology.org/formats/oboInOwl#hasExactSynonym')
    oboInOwl_hasRelatedSynonym = URIRef('http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym')
    obo_hasDefinition = URIRef('http://purl.obolibrary.org/obo/IAO_0000115')
    oboInOwl_hasDbXref = URIRef('http://www.geneontology.org/formats/oboInOwl#hasDbXref')

    obo_part_of = URIRef('http://data.bioontology.org/metadata/obo/part_of')
    pw_part_of = URIRef('http://purl.obolibrary.org/obo/pw#part_of')

    def __init__(self,
                 name: str,
                 filename: str = None) -> None:
        paths = PathhierPaths()
        self.name = name
        if filename:
            self.filename = filename
        else:
            self.filename = paths.pathway_ontology_file
        self.ns = dict()
        self.graph = Graph()

    def load_from_file(self) -> None:
        """
        Load ontology from source
        :param loc:
        :return:
        """
        assert(os.path.exists(self.filename))
        self.graph.load(self.filename)

        for prefix, uri in self.graph.namespaces():
            self.ns[prefix] = uri

        print("Finished loading %s" % self.filename)
        print("Number of entities: %i" % len(self.graph.all_nodes()))

    @property
    def owl_classes(self):
        """
        Generator for owl classes
        :return:
        """
        for cl in self.graph.subjects(RDF.type, OWL.Class):
            yield cl

    @property
    def owl_annotation_properties(self):
        """
        Generator for owl annotation properties
        :return:
        """
        for ap in self.graph.subjects(RDF.type, OWL.AnnotationProperty):
            yield ap

    @property
    def owl_object_properties(self):
        """
        Generator for owl object properties
        :return:
        """
        for op in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            yield op

    def get_label(self, uri) -> str:
        """
        Get label of object given by uri
        :param uri
        :return:
        """
        label = self.graph.label(uri)
        if label:
            return label.value
        return None

    def get_all_labels(self, uri) -> List:
        """
        Get all labels of object given by uri
        :param uri:
        :return:
        """
        labels = []
        for lbl_type, lbl_value in self.graph.preferredLabel(uri):
            labels.append(lbl_value.value)
        return labels

    def get_synonyms(self, uri) -> List:
        """
        Get synonyms of object given by uri
        :param uri:
        :return:
        """
        synonyms = []
        for syn in self.graph.objects(uri, self.oboInOwl_hasExactSynonym):
            synonyms.append(syn.value)
        for syn in self.graph.objects(uri, self.oboInOwl_hasRelatedSynonym):
            synonyms.append(syn.value)
        return synonyms

    def get_definition(self, uri) -> List:
        """
        Get definition of object given by uri
        :param uri:
        :return:
        """
        definitions = []
        for definition in self.graph.objects(uri, self.obo_hasDefinition):
            definitions.append(definition.value)
        for definition in self.graph.objects(uri, RDFS.comment):
            definitions.append(definition.value)
        return definitions

    def get_xrefs(self, uri) -> List:
        """
        Get xrefs of object given by uri
        :param uri:
        :return:
        """
        xrefs = []
        for xref in self.graph.objects(uri, self.oboInOwl_hasDbXref):
            xrefs.append(xref.value)
        return xrefs

    def get_subClassOf(self, uri) -> List:
        """
        Get superclass of obj
        :param uri:
        :return:
        """
        superclasses = []
        for sc in self.graph.objects(uri, RDFS.subClassOf):
            superclasses.append(sc)
        return superclasses

    def get_part_of(self, uri) -> List:
        """
        Get part superclasses of obj
        :param uri:
        :return:
        """
        part_super = []
        for ps in self.graph.objects(uri, self.obo_part_of):
            part_super.append(ps)
        for ps in self.graph.objects(uri, self.pw_part_of):
            part_super.append(ps)
        return part_super


