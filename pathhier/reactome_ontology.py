from overrides import overrides
from typing import List

from rdflib import Namespace
from rdflib.namespace import RDF, RDFS

from pathhier.ontology import Ontology


BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")

# class for representing reactome ontology (has BP3 properties and types)
class ReactomeOntology(Ontology):
    def __init__(self,
                 name: str,
                 filename: str = None):
        super().__init__(name, filename)

    @overrides
    def get_label(self, uri) -> str:
        """
        Get label of object given by uri
        :param uri
        :return:
        """
        label = self.graph.label(uri)
        if label:
            return label.value

        displayNames = self.graph.objects(uri, BP3['displayName'])
        if displayNames:
            return list(displayNames)[0].value

        standardNames = self.graph.objects(uri, BP3['standardName'])
        if standardNames:
            return list(standardNames)[0].value

        return ""

    @overrides
    def get_all_labels(self, uri) -> List:
        """
        Get preferred label of object given by uri
        :param uri:
        :return:
        """
        labels = []
        for lbl_type, lbl_value in self.graph.preferredLabel(uri):
            labels.append(lbl_value.value)
        for lbl_value in self.graph.objects(uri, BP3['displayName']):
            labels.append(lbl_value.value)
        for lbl_value in self.graph.objects(uri, BP3['standardName']):
            labels.append(lbl_value.value)
        for lbl_value in self.graph.objects(uri, BP3['name']):
            labels.append(lbl_value.value)
        return labels

    @overrides
    def get_xrefs(self, uri) -> List:
        """
        Get xrefs of object given by uri
        :param uri:
        :return:
        """
        xrefs = []
        for xref in self.graph.objects(uri, self.oboInOwl_hasDbXref):
            xrefs.append(xref.value)
        for xref in self.graph.objects(uri, BP3['xref']):
            db = self.graph.objects(xref, BP3['db'])
            id = self.graph.objects(xref, BP3['id'])
            if db and id:
                x_val = '{}:{}'.format(db[0].value, id[0].value)
                xrefs.append(x_val)
        for entRef in self.graph.objects(uri, BP3['entityReference']):
            xref_entries = self.graph.objects(entRef, BP3['xref'])
            for xref in xref_entries:
                db = self.graph.objects(xref, BP3['db'])
                id = self.graph.objects(xref, BP3['id'])
                if db and id:
                    x_val = '{}:{}'.format(db[0].value, id[0].value)
                    xrefs.append(x_val)
        return xrefs

    @overrides
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

    @overrides
    def get_part_of(self, uri) -> List:
        """
        Get part superclasses of obj
        :param uri:
        :return:
        """
        part_super = []
        for ps in self.graph.objects(uri, self.obo_part_of):
            part_super.append(ps)
        for ps in self.graph.subjects(BP3['pathwayComponent'], uri):
            if (ps, RDF.type, BP3['Pathway']) in self.graph:
                part_super.append(ps)
        return part_super
