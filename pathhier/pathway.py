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
        self.lid = 0
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

    def to_json(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "aliases": self.aliases,
            "xrefs": self.xrefs,
            "definition": self.definition,
            "obj_type": self.obj_type
        }


# class for representing a complex (entity that has components which are other entities)
class Complex(Entity):
    def __init__(self,
                 uid: str,
                 name: str,
                 components: List[Entity]) -> None:
        self.uid = uid
        self.name = name
        self.aliases = []
        self.definition = ''
        self.xrefs = []
        self.components = components
        self.obj_type = 'Complex'
        super(Entity, self).__init__()

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "name": self.name,
                "components": [ent.name for ent in self.components]
            }
        )

    def to_json(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "components": [ent.name for ent in self.components]
        }


# class for representing a group (entity that has members which are other entities)
class Group(Entity):
    def __init__(self,
                 uid: str,
                 name: str,
                 members: List[Entity]) -> None:
        self.uid = uid
        self.name = name
        self.aliases = []
        self.definition = ''
        self.xrefs = []
        self.members = members
        self.obj_type = 'Group'
        super(Entity, self).__init__()

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "name": self.name,
                "members": [ent.name for ent in self.members]
            }
        )

    def to_json(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "components": [ent.name for ent in self.members]
        }


# class for representing a reaction (entity that has left, right, modifier, and participant entities)
class Reaction(Entity):
    def __init__(self,
                 uid: str,
                 name: str,
                 left: List[Entity],
                 right: List[Entity],
                 controllers: List[Entity],
                 other: List[Entity]) -> None:
        self.uid = uid
        self.name = name
        self.aliases = []
        self.definition = ''
        self.xrefs = []
        self.left = left
        self.right = right
        self.controllers = controllers
        self.other = other
        self.obj_type = 'Reaction'
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

    def to_json(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "aliases": self.aliases,
            "xrefs": self.xrefs,
            "definition": self.definition,
            "obj_type": self.obj_type,
            "left": [ent.name for ent in self.left],
            "right": [ent.name for ent in self.right],
            "controllers": [ent.name for ent in self.controllers],
            "other": [ent.name for ent in self.other]
        }


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
                "comment": self.comments,
                "xrefs": self.xrefs
            },
            sort_keys=True,
            indent=4
        )

    def get_entity_by_uid(self, uid):
        """
        Get entity by URI
        :param uri:
        :return:
        """
        matches = [ent for ent in self.entities if ent.uid == uid]
        if matches:
            return matches[0]
        else:
            return None

    def get_all_complex_xrefs(self, cx):
        """
        Get all xrefs of complex
        :param cx:
        :return:
        """
        all_xrefs = []
        if cx:
            all_xrefs += cx.xrefs

            if cx.obj_type == 'Complex':
                for comp in cx.components:
                    all_xrefs += self.get_all_complex_xrefs(self.get_entity_by_uid(comp))
        return all_xrefs


# class for representing a pathway KB
class PathKB:
    def __init__(self, name: str, loc: str = None):
        self.name = name
        self.loc = loc
        self.uid_to_pathway_dict = dict()
        self.name_to_pathway_dict = defaultdict(list)
        self.xref_to_pathway_dict = defaultdict(list)
        self.pathways = []
        self.hierarchy = []

    def _construct_lookup_dicts(self):
        """for
        Construct lookup dicts for pathways based on pathway id, names, and xrefs
        :return:
        """
        self.uid_to_pathway_dict.clear()
        self.name_to_pathway_dict.clear()
        self.xref_to_pathway_dict.clear()

        for i, p in enumerate(self.pathways):
            if p.uid in self.uid_to_pathway_dict:
                print("Skipped: %s already in pathway list" % p.uid)
                continue
            self.uid_to_pathway_dict[p.uid] = i
            self.name_to_pathway_dict[p.name].append(i)
            for x in p.xrefs:
                self.xref_to_pathway_dict[x].append(i)
        return

    def get_pathway_by_uid(self, uid: str):
        """
        Returns pathway with matching pathway uid
        :param uid:
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
            all_names += [n.value for n in list(g.objects(ent_uid, BP3[name_prop])) if not(n.endswith('...'))]
        return list(set(all_names))

    def _get_all_ent_xrefs(self, ent_refs, g, done_list):
        """
        Iterate through ent_refs and get xrefs
        :param ent_refs:
        :param g:
        :param done_list:
        :return:
        """
        all_xrefs = []
        next_refs = []

        for ref in ent_refs:
            for xobj in g.objects(ref, BP3.xref):
                if (xobj, RDF.type, BP3.UnificationXref) in g:
                    db = list(g.objects(xobj, BP3.db))
                    id = list(g.objects(xobj, BP3.id))
                    if db and id:
                        xref_id = "{}:{}".format(db[0], id[0])
                        all_xrefs.append(xref_id)
                elif (xobj, RDF.type, BP3.ProteinReference) in g \
                    or (xobj, RDF.type, BP3.SmallMoleculeReference) in g \
                    or (xobj, RDF.type, BP3.RnaReference) in g \
                    or (xobj, RDF.type, BP3.DnaReference) in g \
                    or (xobj, RDF.type, BP3.DnaReference) in g:
                    next_refs += list(g.objects(ref, BP3["entityReference"]))
                else:
                    all_xrefs.append(str(xobj))
            done_list.add(ref)

        if next_refs:
            all_xrefs += self._get_all_ent_xrefs(list(set(next_refs).difference(set(done_list))), g, done_list)

        return all_xrefs

    def _get_all_mem_xrefs(self, mem_refs, g, done_list):
        """
        Iterate through mem_refs and get xrefs
        :param mem_refs:
        :param g:
        :param done_list:
        :return:
        """
        all_xrefs = []
        next_refs = []

        for ref in mem_refs:
            for xobj in g.objects(ref, BP3.xref):
                if (xobj, RDF.type, BP3.UnificationXref) in g:
                    db = list(g.objects(xobj, BP3.db))
                    id = list(g.objects(xobj, BP3.id))
                    if db and id:
                        xref_id = "{}:{}".format(db[0], id[0])
                        all_xrefs.append(xref_id)
                elif (xobj, RDF.type, BP3.ProteinReference) in g \
                        or (xobj, RDF.type, BP3.SmallMoleculeReference) in g \
                        or (xobj, RDF.type, BP3.RnaReference) in g \
                        or (xobj, RDF.type, BP3.DnaReference) in g \
                        or (xobj, RDF.type, BP3.DnaReference) in g:
                    next_refs += list(g.objects(ref, BP3["entityReference"])) \
                                 + list(g.objects(ref, BP3["memberEntityReference"])) \
                                 + list(g.objects(ref, BP3["memberPhysicalEntity"]))
                else:
                    all_xrefs.append(str(xobj))

            # special case for panther
            for xobj in g.objects(ref, BP3["memberEntityReference"]):
                if 'uniprot' in xobj:
                    id = xobj.split('/')[-1]
                    all_xrefs.append("{}:{}".format("Uniprot", id))

            done_list.add(ref)

        if next_refs:
            all_xrefs += self._get_all_mem_xrefs(list(set(next_refs).difference(set(done_list))), g, done_list)

        return all_xrefs

    def _get_biopax_xrefs(self, ent_uid, g):
        """
        Get all xrefs of entity from graph g
        :param ent_uid:
        :param g:
        :return:
        """
        ent_refs = list(g.objects(ent_uid, BP3["entityReference"])) \
            + [ent_uid]

        mem_ent_refs = list(g.objects(ent_uid, BP3["memberEntityReference"])) \
            + list(g.objects(ent_uid, BP3["memberPhysicalEntity"]))

        all_ent_xrefs = self._get_all_ent_xrefs(ent_refs, g, set([]))
        all_mem_xrefs = self._get_all_mem_xrefs(mem_ent_refs, g, set([]))

        return pathway_utils.clean_xrefs(
                   all_ent_xrefs,
                   constants.PATHWAY_XREF_AVOID_TERMS
               ), pathway_utils.clean_xrefs(
                   all_mem_xrefs,
                   constants.PATHWAY_XREF_AVOID_TERMS
               )

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

        xrefs, mem_xrefs = self._get_biopax_xrefs(ent_uid, g)

        if mem_xrefs:
            grp = Group(
                uid=ent_uid,
                name=ent_names[0],
                members=mem_xrefs
            )
            grp.aliases = ent_names
            grp.xrefs = xrefs
            grp.definition = self._get_biopax_definition(ent_uid, g)
            grp.obj_type = 'Group'
            return grp
        else:
            return Entity(
                uid=ent_uid,
                name=ent_names[0],
                aliases=ent_names,
                xrefs=xrefs,
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
        component_entities = list(g.objects(cx_uid, BP3['memberPhysicalEntity'])) \
                     + list(g.objects(cx_uid, BP3['component']))
        components = []

        for ent in component_entities:
            ent_type = str(list(g.objects(ent, RDF.type))[0]).split('#')[-1]
            if ent_type == "Pathway":
                pass
            elif ent_type == "Complex":
                components += self._process_biopax_complex(ent, g)
            else:
                components.append(self._process_biopax_entity(ent, ent_type, g))

        cx_names = self._get_biopax_names(cx_uid, g)

        complex_object = Complex(
            uid=cx_uid,
            name=cx_names[0],
            components=components
        )

        complex_object.aliases = cx_names
        complex_object.xrefs, _ = self._get_biopax_xrefs(cx_uid, g)
        complex_object.definition = self._get_biopax_definition(cx_uid, g)
        complex_object.obj_type = "Complex"

        return [complex_object] + components

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
                print('Warning: reaction nesting...')
                entities += self._process_biopax_reaction(ent, ent_type, g)
            elif ent_type == "Complex":
                entities += self._process_biopax_complex(ent, g)
            else:
                entities.append(self._process_biopax_entity(ent, ent_type, g))

        rx_names = self._get_biopax_names(rx_uid, g)

        if rx_names:
            rx_name = rx_names[0]
            rx_aliases = rx_names
        elif self.name == "pid":
            comments = self._get_biopax_comments(rx_uid, g)
            for com in comments:
                if com.startswith("REPLACED"):
                    rx_name = rx_uid
                    rx_aliases = [rx_uid, com.split('_')[-1]]
        else:
            print("No name: %s" % rx_uid)
            rx_name = rx_uid
            rx_aliases = [rx_uid]

        reaction_object = Reaction(
            uid=rx_uid,
            name=rx_name,
            left=[ent for ent in entities if ent.uid in left],
            right=[ent for ent in entities if ent.uid in right],
            controllers=[ent for ent in entities if ent.uid in controllers],
            other=[ent for ent in entities if ent.uid in other]
        )

        reaction_object.aliases = rx_aliases

        reaction_object.xrefs, _ = self._get_biopax_xrefs(rx_uid, g)
        reaction_object.definition = self._get_biopax_definition(rx_uid, g)
        reaction_object.obj_type = rx_type

        return [reaction_object] + entities

    def _process_biopax_pathway(self, pathway_uid, g):
        """
        Construct a pathway object from pathway in graph g
        :param pathway_uid: pathway
        :param g: biopax graph
        :return:
        """

        # get UID from HumanCyc
        def get_uid_humancyc(uid):
            xrefs, _ = self._get_biopax_xrefs(uid, g)
            humancyc_id = [xref.split(':')[-1] for xref in xrefs if xref.split(':')[0] == 'HumanCyc']
            if humancyc_id:
                return '{}:{}'.format('HumanCyc', humancyc_id[0])
            return uid

        # get UID from Reactome
        def get_uid_reactome(uid):
            xrefs, _ = self._get_biopax_xrefs(uid, g)
            reactome_id = [xref.split(':')[-1] for xref in xrefs if xref.split(':')[0] == 'Reactome']
            if reactome_id:
                return '{}:{}'.format('Reactome', reactome_id[0])
            return uid

        # get UID from SMPDB
        def get_uid_smpdb(uid):
            xrefs, _ = self._get_biopax_xrefs(uid, g)
            smpdb_id = [xref.split(':')[-1] for xref in xrefs if xref.split(':')[0] == 'SMPDB']
            if len(smpdb_id) == 1:
                return smpdb_id[0].split('/')[-1]
            return uid

        # get UID from pathway commons PID KB
        def get_uid_pid(uid):
            comments = self._get_biopax_comments(uid, g)
            for com in comments:
                if com.startswith('REPLACED'):
                    uid = '{}:{}'.format(self.name, com.split('_')[-1])
                    return uid
            return uid

        # clean URIs from BioModels
        def clean_uri_biomodels(ent):
            ent.uid = ent.uid.split('other_data/')[-1]
            return ent

        # get pathway namess
        pathway_names = self._get_biopax_names(pathway_uid, g)

        # initialize pathway variables
        pathway_subpaths = set([])
        pathway_entities = []
        pathway_relations = []

        # process each pathway component
        for component_uid in list(g.objects(pathway_uid, BP3["pathwayComponent"])):
            # get component type
            comp_type = str(list(g.objects(component_uid, RDF.type))[0]).split('#')[-1]

            # check if pathway -> subpaths
            if comp_type == "Pathway":
                if self.name == "smpdb":
                    pathway_subpaths.add(get_uid_smpdb(component_uid))
                else:
                    pathway_subpaths.add(component_uid)
            # else process entity
            elif comp_type in constants.BIOPAX_RX_TYPES:
                pathway_entities += self._process_biopax_reaction(component_uid, comp_type, g)
            elif comp_type == "Complex":
                pathway_entities += self._process_biopax_complex(component_uid, g)
            else:
                pathway_entities.append(self._process_biopax_entity(component_uid, comp_type, g))

        # process entities into relations
        for ent in pathway_entities:
            if ent.obj_type in constants.BIOPAX_RX_TYPES:
                for left in ent.left:
                    pathway_relations.append((ent.uid, 'participant', left.uid))
                for right in ent.right:
                    pathway_relations.append((ent.uid, 'participant', right.uid))
                for controller in ent.controllers:
                    pathway_relations.append((ent.uid, 'controller', controller.uid))
                for other in ent.other:
                    pathway_relations.append((ent.uid, 'other', other.uid))
            elif ent.obj_type == 'Complex':
                for comp in ent.components:
                    if type(comp) == Entity:
                        pathway_relations.append((ent.uid, 'component', comp.uid))
            elif ent.obj_type == 'Group':
                for mem in ent.members:
                    if type(mem) == Entity:
                        pathway_relations.append((ent.uid, 'member', mem.uid))

        xrefs, _ = self._get_biopax_xrefs(pathway_uid, g)

        # kb-specific processing
        if self.name != "kegg":
            pathway_subpaths = pathway_utils.clean_subpaths(self.name, pathway_subpaths)

        if self.name == "humancyc":
            pathway_uid = get_uid_humancyc(pathway_uid)
        elif self.name == "reactome":
            pathway_uid = get_uid_reactome(pathway_uid)
        elif self.name == "smpdb":
            pathway_uid = get_uid_smpdb(pathway_uid)
        elif self.name == "pid":
            pathway_uid = get_uid_pid(pathway_uid)
        else:
            pathway_uid = pathway_utils.clean_path_id(self.name, pathway_uid)

        if self.name == "biomodels":
            pathway_entities = [clean_uri_biomodels(ent) for ent in pathway_entities]

        # set pathway name to URI if no name
        if len(pathway_names) == 0:
            pathway_name = pathway_uid
        else:
            pathway_name = pathway_names[0]

        pathway_object = Pathway(
            uid=pathway_uid,
            name=pathway_name,
            aliases=pathway_names,
            xrefs=pathway_utils.clean_xrefs(xrefs, constants.PATHWAY_XREF_AVOID_TERMS),
            definition=self._get_biopax_definition(pathway_uid, g),
            comments=self._get_biopax_comments(pathway_uid, g),
            subpaths=pathway_subpaths,
            entities=pathway_entities,
            relations=pathway_relations,
            provenance=self.name
        )

        return pathway_object

    def _extract_pathway_hierarchy(self, g) -> None:
        """
        Extract pathway hierarchy from kb
        :param g: graph of KB
        :return:
        """
        for s, p, o in g:
            if (s, RDF.type, BP3['Pathway']) in g \
                    and (o, RDF.type, BP3['Pathway']) in g:
                if p == BP3['pathwayComponent']:
                    self.hierarchy.append((s, o))
                elif p == RDF.type:
                    self.hierarchy.append((o, s))

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
            if self.name != "smpdb" or (self.name == "smpdb" and "SubPathways" not in pathway_uid):
                pathways.append(self._process_biopax_pathway(pathway_uid, g))

        self._extract_pathway_hierarchy(g)

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

        # initialize variables
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

        graph_id_dict = dict()

        # iterate through graph ID labels
        for label in root.findall('pv:Label', ns):
            if ('TextLabel' in label.attrib) and ('GraphId' in label.attrib):
                label_text = label.attrib['TextLabel']
                graph_id = label.attrib['GraphId']
                group_ref = None
                if 'GroupRef' in label.attrib:
                    group_ref = label.attrib['GroupRef']

                graph_id_dict[graph_id] = {
                    'label': label_text,
                    'group_ref': group_ref,
                    'entity': None
                }

        # iterate through groups
        for group in root.findall('pv:Group', ns):
            group_id = group.attrib['GroupId']
            if 'GraphId' in group.attrib:
                graph_id = group.attrib['GraphId']
                if graph_id in graph_id_dict:
                    graph_id_dict[graph_id]['group_ref'] = group_id
                else:
                    graph_id_dict[graph_id] = {
                        'label': group_id,
                        'group_ref': group_id,
                        'entity': None
                    }

        # iterate through data nodes
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

                if ent_type in constants.ENT_TYPE_MAP:
                    ent_type = constants.ENT_TYPE_MAP[ent_type]

                new_ent = Entity(
                    uid=ent_name,
                    name=ent_name,
                    aliases=[ent_name],
                    xrefs=pathway_utils.clean_xrefs(ent_xrefs, constants.ENTITY_XREF_AVOID_TERMS),
                    definition='',
                    obj_type=ent_type
                )

                if 'GraphId' in child.attrib:
                    group_ref = None
                    if 'GroupRef' in child.attrib:
                        group_ref = child.attrib['GroupRef']

                    graph_id_dict[child.attrib['GraphId']] = {
                        'label': ent_name,
                        'group_ref': group_ref,
                        'entity': new_ent
                    }

                pathway_entities.append(new_ent)

        # add membership relations
        group_ents = defaultdict(list)
        group_names = dict()

        for graph_id, graph_props in graph_id_dict.items():

            label = graph_props['label']
            group_ref = graph_props['group_ref']
            entity = graph_props['entity']

            if group_ref and label:
                if entity:
                    group_ents[group_ref].append(entity)
                else:
                    if label != group_ref:
                        group_names[group_ref] = label

        # add group entities to pathway entities
        groups = []

        for group_ref, group_members in group_ents.items():
            group_name = group_ref
            if group_ref in group_names:
                group_name = group_names[group_ref]

            new_group = Group(
                uid=group_ref,
                name=group_name,
                members=group_members
            )

            groups.append(new_group)

        pathway_entities += groups

        for grp in groups:
            for mem in grp.members:
                pathway_relations.append((grp.uid, 'member', mem.uid))

        # add interactions
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

                        if arrowhead in constants.WP_PROPERTIES:
                            relation = constants.WP_PROPERTIES[arrowhead]
                        else:
                            relation = 'other'

                        origin = graphref1
                        target = graphref2

                        if graphref1 in graph_id_dict:
                            origin = graph_id_dict[graphref1]['entity']
                            if not origin:
                                origin_group_ref = graph_id_dict[graphref1]['group_ref']
                                origin_matches = [g for g in groups if g.uid == origin_group_ref]
                                if origin_matches:
                                    origin = origin_matches[0].uid
                                else:
                                    origin = graph_id_dict[graphref1]['label']
                            else:
                                if type(origin) == Entity:
                                    origin = origin.uid

                        if graphref2 in graph_id_dict:
                            target = graph_id_dict[graphref2]['entity']
                            if not target:
                                target_group_ref = graph_id_dict[graphref2]['group_ref']
                                target_matches = [g for g in groups if g.uid == target_group_ref]
                                if target_matches:
                                    target = target_matches[0].uid
                                else:
                                    target = graph_id_dict[graphref2]['label']
                            else:
                                if type(target) == Entity:
                                    target = target.uid

                        if origin and target:
                            pathway_relations.append((origin, relation, target))

        pathway = Pathway(
            uid=pathway_uid,
            name=pathway_name,
            aliases=pathway_aliases,
            xrefs=pathway_utils.clean_xrefs(pathway_xrefs, constants.PATHWAY_XREF_AVOID_TERMS),
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





