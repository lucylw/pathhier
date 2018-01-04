import os
import sys
import json
import glob
import tqdm
import itertools
import pickle
from typing import List, Set
from collections import defaultdict
from lxml import etree

from rdflib import Graph
from rdflib import Namespace
from rdflib.namespace import RDF

import pathhier.constants as constants
import pathhier.utils.pathway_utils as pathway_utils


RDF_EXTS = ['.owl', '.OWL', '.rdf', '.RDF', '.ttl', '.TTL']
GPML_EXTS = ['.gpml', '.GPML']
SBML_EXTS = ['.sbml', '.SBML']

BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")


# class for representing an entity
class Entity:
    def __init__(self,
                 uid: str,
                 name: str,
                 aliases: List[str],
                 xrefs: List[str],
                 definition: str,
                 obj_type: str) -> None:
        self.uid = uid
        self.name = name
        self.aliases = aliases
        self.xrefs = xrefs
        self.definition = definition
        self.obj_type = obj_type

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "name": self.name,
                "xrefs": self.xrefs
            }
        )

    def __eq__(self, ent_id: str):
        return self.uid == ent_id


# class for representing a complex (entity that has components which are other entities)
class Complex(Entity):
    def __init__(self,
                 components: List[Entity]) -> None:
        super(Entity, self).__init__()
        self.components = components


# class for representing a reaction (entity that has left, right, modifier, and participant entities)
class Reaction(Entity):
    def __init__(self,
                 left: List[Entity],
                 right: List[Entity],
                 controllers: List[Entity],
                 other: List[Entity]) -> None:
        self.left = left
        self.right = right
        self.controllers = controllers
        self.other = other
        super(Entity, self).__init__()

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "name": self.name,
                "left": [ent.name for ent in self.left],
                "right": [ent.name for ent in self.right],
                "controllers": [ent.name for ent in self.controllers],
                "other": [ent.name for ent in self.other]
            }
        )

    def get_participants(self):
        """
        Get all participants of reaction
        :return:
        """
        for ent in itertools.chain(self.left, self.right, self.controllers, self.other):
            yield ent


# class for representing a pathway
class Pathway:
    def __init__(self,
                 uid: str,
                 name: str,
                 aliases: List[str],
                 xrefs: List[str],
                 definition: str,
                 comments: List[str],
                 subpaths: Set[str],    # set of pathway uids
                 entities: List[Entity],
                 relations: List[tuple],
                 provenance: str
                 ):
        self.uid = uid
        self.name = name
        self.aliases = aliases
        self.xrefs = xrefs
        self.definition = definition
        self.comments = comments
        self.subpaths = subpaths
        self.entities = entities
        self.relations = relations
        self.provenance = provenance

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "name": self.name,
                "definition": self.definition,
                "xrefs": self.xrefs
            }
        )


# class for representing a pathway KB
class PathKB:
    def __init__(self, name: str, loc: str = None):
        self.name = name
        self.loc = loc
        self.uid_to_pathway_dict = dict()
        self.name_to_pathway_dict = defaultdict(list)
        self.xref_to_pathway_dict = defaultdict(list)
        self.pathways = []

    def _construct_lookup_dicts(self):
        """for
        Construct lookup dicts for pathways based on pathway id, names, and xrefs
        :return:
        """
        self.uid_to_pathway_dict.clear()
        self.name_to_pathway_dict.clear()
        self.xref_to_pathway_dict.clear()

        for i, p in enumerate(self.pathways):
            assert p.uid not in self.uid_to_pathway_dict
            self.uid_to_pathway_dict[p.uid] = i
            self.name_to_pathway_dict[p.name].append(i)
            for x in p.xrefs:
                self.xref_to_pathway_dict[x].append(i)
        return

    def get_pathway_by_uid(self, uid: str):
        """
        Returns pathway with matching pathway uid
        :param pid:
        :return:
        """
        if uid in self.uid_to_pathway_dict:
            return self.pathways[self.uid_to_pathway_dict[uid]]
        else:
            return None

    def get_pathway_by_name(self, name: str):
        """
        Returns pathways with matching name
        :param name:
        :return:
        """
        return [self.pathways[i] for i in self.name_to_pathway_dict[name]]

    def get_pathway_by_xref(self, xref: str):
        """
        Returns pathways with matching xref
        :param xref:
        :return:
        """
        return [self.pathways[i] for i in self.xref_to_pathway_dict[xref]]

    @staticmethod
    def _get_biopax_names(ent_uid, g):
        """
        Get all names of entity from graph g
        :param ent_uid:
        :param g:
        :return:
        """
        all_names = []
        for name_prop in constants.BIOPAX_NAME_PROPERTIES:
            all_names += [n.value for n in list(g.objects(ent_uid, BP3[name_prop]))]
        return all_names

    def _get_biopax_xrefs(self, ent_uid, g):
        """
        Get all xrefs of entity from graph g
        :param ent_uid:
        :param g:
        :return:
        """
        all_xrefs = []
        for xobj in g.objects(ent_uid, BP3["xref"]):
            if (xobj, RDF.type, BP3.UnificationXref) in g:
                db = list(g.objects(xobj, BP3["db"]))[0]
                id = list(g.objects(xobj, BP3["id"]))[0]
                xref_id = "{}:{}".format(db, id)
                all_xrefs.append(xref_id)
            elif (xobj, RDF.type, BP3.ProteinReference) in g \
                or (xobj, RDF.type, BP3.SmallMoleculeReference) in g \
                or (xobj, RDF.type, BP3.RnaReference) in g \
                or (xobj, RDF.type, BP3.DnaReference) in g:
                all_xrefs += self._get_biopax_xrefs(xobj, g)
        return pathway_utils.clean_xrefs(all_xrefs)

    @staticmethod
    def _get_biopax_definition(ent_uid, g):
        """
        Get definition of entity from graph g
        :param ent_uid:
        :param g:
        :return:
        """
        definition = list(g.objects(ent_uid, BP3["definition"]))
        if definition:
            return definition[0].value
        else:
            return ""

    @staticmethod
    def _get_biopax_comments(ent_uid, g):
        """
        Get comments of entity from graph g
        :param ent_uid:
        :param g:
        :return:
        """
        return [c.value for c in list(g.objects(ent_uid, BP3["comment"]))]

    def _process_biopax_entity(self, ent_uid, ent_type, g):
        """
        Process biopax entity (small molecule, protein, rna, dna etc)
        :param ent_uid:
        :param ent_type:
        :param g:
        :return:
        """
        ent_names = self._get_biopax_names(ent_uid, g)

        if not ent_names:
            ent_names = [ent_uid]

        return Entity(
            uid=ent_uid,
            name=ent_names[0],
            aliases=ent_names[1:],
            xrefs=self._get_biopax_xrefs(ent_uid, g),
            definition=self._get_biopax_definition(ent_uid, g),
            obj_type=ent_type
        )

    def _process_biopax_complex(self, cx_uid, g):
        """
        Process biopax complex
        :param cx_uid:
        :param g:
        :return:
        """
        # initialize entities
        entities = []

        components = list(g.objects(cx_uid, BP3['component']))

        for ent in components:
            ent_type = str(list(g.objects(ent, RDF.type))[0]).split('#')[-1]
            if ent_type == "Pathway":
                pass
            elif ent_type in constants.BIOPAX_RX_TYPES:
                entities += self._process_biopax_reaction(ent, ent_type, g)
            elif ent_type == "Complex":
                entities += self._process_biopax_complex(ent, g)
            else:
                entities.append(self._process_biopax_entity(ent, ent_type, g))

        complex_object = Complex(
            components=components
        )

        cx_names = self._get_biopax_names(cx_uid, g)
        complex_object.uid = cx_uid
        complex_object.name = cx_names[0]
        complex_object.aliases = cx_names[1:]
        complex_object.xrefs = self._get_biopax_xrefs(cx_uid, g)
        complex_object.definition = self._get_biopax_definition(cx_uid, g)
        complex_object.obj_type = "Complex"

        entities.append(complex_object)

        return entities

    def _process_biopax_reaction(self, rx_uid, rx_type, g):
        """
        Process biopax reaction
        :param rx_uid:
        :param rx_type: type of reaction
        :param g:
        :return:
        """
        # initialize entities
        entities = []

        # get all objects
        left = list(g.objects(rx_uid, BP3["left"]))
        right = list(g.objects(rx_uid, BP3["right"])) + list(g.objects(rx_uid, BP3["product"]))
        other = list(g.objects(rx_uid, BP3["participant"]))

        controllers = []
        control_objs = g.subjects(BP3["controlled"], rx_uid)
        for c in control_objs:
            controllers += list(g.objects(c, BP3["controller"]))

        for ent in itertools.chain(left, right, other, controllers):
            ent_type = str(list(g.objects(ent, RDF.type))[0]).split('#')[-1]
            if ent_type == "Pathway":
                pass
            elif ent_type in constants.BIOPAX_RX_TYPES:
                entities += self._process_biopax_reaction(ent, ent_type, g)
            elif ent_type == "Complex":
                entities += self._process_biopax_complex(ent, g)
            else:
                entities.append(self._process_biopax_entity(ent, ent_type, g))

        reaction_object = Reaction(
            left=[ent for ent in entities if ent.uid in left],
            right=[ent for ent in entities if ent.uid in right],
            controllers=[ent for ent in entities if ent.uid in controllers],
            other=[ent for ent in entities if ent.uid in other]
        )


        reaction_object.uid = rx_uid

        rx_names = self._get_biopax_names(rx_uid, g)
        if rx_names:
            reaction_object.name = rx_names[0]
            reaction_object.aliases = rx_names[1:]
        else:
            print("No name: %s" % rx_uid)
            reaction_object.name = rx_uid
            reaction_object.aliases = []

        reaction_object.xrefs = self._get_biopax_xrefs(rx_uid, g)
        reaction_object.definition = self._get_biopax_definition(rx_uid, g)
        reaction_object.obj_type = rx_type

        entities.append(reaction_object)

        return entities

    def _process_biopax_pathway(self, pathway_uid, g):
        """
        Construct a pathway object from pathway in graph g
        :param pathway_uid: pathway
        :param g: biopax graph
        :return:
        """
        # get pathway data components
        pathway_names = self._get_biopax_names(pathway_uid, g)

        pathway_subpaths = set([])
        pathway_entities = []
        pathway_relations = []

        for component_uid in list(g.objects(pathway_uid, BP3["pathwayComponent"])):
            # get component type
            comp_type = str(list(g.objects(component_uid, RDF.type))[0]).split('#')[-1]

            # check if pathway -> subpaths
            if comp_type == "Pathway":
                pathway_subpaths.add(component_uid)
            # else process entity
            elif comp_type in constants.BIOPAX_RX_TYPES:
                pathway_entities += self._process_biopax_reaction(component_uid, comp_type, g)
            elif comp_type == "Complex":
                pathway_entities += self._process_biopax_complex(component_uid, g)
            else:
                pathway_entities.append(self._process_biopax_entity(component_uid, comp_type, g))

        if self.name != "kegg":
            pathway_subpaths = pathway_utils.clean_subpaths(self.name, pathway_subpaths)

        pathway_object = Pathway(
            uid=pathway_utils.clean_path_id(self.name, pathway_uid),
            name=pathway_names[0],
            aliases=pathway_names[1:],
            xrefs=self._get_biopax_xrefs(pathway_uid, g),
            definition=self._get_biopax_definition(pathway_uid, g),
            comments=self._get_biopax_comments(pathway_uid, g),
            subpaths=pathway_subpaths,
            entities=pathway_entities,
            relations=pathway_relations,
            provenance=self.name
        )
        return pathway_object

    def _load_from_biopax(self, loc):
        """
        Loads pathway from BioPAX file
        :param loc: location of file
        :return:
        """
        # create rdflib Graph and load file
        g = Graph()
        g.load(loc)

        # bind namespaces
        g.bind("rdf", RDF)
        g.bind("bp3", BP3)

        # initialize
        pathways = []
        pathway_list = list(g.subjects(RDF.type, BP3["Pathway"]))

        for pathway_uid in tqdm.tqdm(pathway_list, total=len(pathway_list)):
            pathways.append(self._process_biopax_pathway(pathway_uid, g))

        return pathways

    def _load_from_gpml(self, loc):
        """
        Loads pathway from GPML file
        :param loc: location of file
        :return:
        """
        # parse the file
        try:
            tree = etree.parse(loc)
        except etree.XMLSyntaxError:
            p = etree.XMLParser(huge_tree=True)
            tree = etree.parse(loc, parser=p)

        root = tree.getroot()
        ns = root.nsmap

        # add pathvisio to namespace
        if "pv" not in ns:
            ns['pv'] = "http://pathvisio.org/GPML/2013a"

        if None in ns:
            del ns[None]

        pathway_uid = loc.split('_')[-2]
        pathway_name = root.attrib['Name']
        pathway_aliases = []
        pathway_xrefs = []
        pathway_definition = ""
        pathway_comments = [
            child.text for child in root.findall('pv:Comment', ns)
            if child.text is not None
        ]

        pathway_subpaths = set([])
        pathway_entities = []
        pathway_relations = []

        graphIdTemp = dict()
        groupIdTemp = defaultdict(list)

        for child in root.findall('pv:DataNode', ns):
            if ('Type' in child.attrib) and ('TextLabel' in child.attrib):
                ent_type = child.attrib['Type']
                ent_name = child.attrib['TextLabel'].strip().replace('\n', ' ')

                ent_xrefs = []

                if ent_type == "Pathway":
                    if len(ent_xrefs) > 0:
                        pathway_subpaths.add(ent_xrefs[0])
                    else:
                        pathway_subpaths.add("PathName:" + ent_name)
                elif ent_type in constants.GPML_ENTITY_TYPES:
                    for subchild in child.findall('pv:Xref', ns):
                        xref = subchild.attrib['Database'].strip().replace('\n', ' ') + ':' \
                               + subchild.attrib['ID'].strip().replace('\n', ' ')
                        if len(xref) > 1:
                            ent_xrefs.append(xref)
                else:
                    sys.stderr.write("%s\n" % loc)
                    sys.stderr.write("Unknown type: %s\n" % ent_type)

                if 'GraphId' in child.attrib:
                    graphIdTemp[child.attrib['GraphId']] = ent_name

                if 'GroupId' in child.attrib:
                    groupIdTemp[child.attrib['GroupId']].append(ent_name)

        for group, members in groupIdTemp.items():
            for m in members:
                pathway_relations.append((group, "member", m))

        for child in root.findall('pv:Interaction', ns):
            for graphic in child.findall('pv:Graphics', ns):
                points = graphic.findall('pv:Point', ns)
                if len(points) == 2:
                    if ('GraphRef' in points[0].attrib) and \
                            ('GraphRef' in points[1].attrib) and \
                            ('ArrowHead' in points[1].attrib):
                        graphref1 = points[0].attrib['GraphRef']
                        graphref2 = points[1].attrib['GraphRef']
                        arrowhead = points[1].attrib['ArrowHead']

                        relation = "controller" if arrowhead == "Arrow" else arrowhead
                        origin = graphIdTemp[graphref1] if graphref1 in graphIdTemp else graphref1
                        target = graphIdTemp[graphref2] if graphref2 in graphIdTemp else graphref2

                        pathway_relations.append((origin, relation, target))

        pathway = Pathway(
            uid=pathway_uid,
            name=pathway_name,
            aliases=pathway_aliases,
            xrefs=pathway_xrefs,
            definition=pathway_definition,
            comments=pathway_comments,
            subpaths=pathway_subpaths,
            entities=pathway_entities,
            relations=pathway_relations,
            provenance=self.name
        )

        return [pathway]

    def _load_from_sbml(self, loc):
        """
        Loads pathway from SBML
        :param loc: location of file
        :return:
        """
        raise(NotImplementedError, "SBML file reader not implemented yet!")

    def load(self, location):
        """
        Sends file to appropriate reader, or iterate through files if given a directory
        :param location:
        :return:
        """
        if os.path.isfile(location):
            fname, fext = os.path.splitext(location)
            if fext in RDF_EXTS:
                return self._load_from_biopax(location)
            elif fext in GPML_EXTS:
                return self._load_from_gpml(location)
            elif fext in SBML_EXTS:
                return self._load_from_sbml(location)
            else:
                raise(NotImplementedError, "Unknown file type! {}".format(location))
        elif os.path.isdir(location):
            files = glob.glob(os.path.join(location, '*.*'))
            for f in tqdm.tqdm(files, total=len(files)):
                self.pathways += self.load(f)
            self._construct_lookup_dicts()
            return

    @staticmethod
    def load_pickle(kb_name, in_path):
        """
        Load pathways from pickle file
        :param kb_name:
        :param in_path:
        :return:
        """
        kb = PathKB(kb_name, in_path)
        with open(in_path, 'rb') as f:
            kb.pathways = pickle.load(f)
            kb._construct_lookup_dicts()
        return kb

    def dump_pickle(self, out_path):
        """
        Dump pathways to pickle file
        :param out_path:
        :return:
        """
        with open(out_path, 'wb') as outf:
            pickle.dump(list(self.pathways), outf)





