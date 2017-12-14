import os
import sys
import json
import glob
import networkx as nx

from rdflib import Graph
from rdflib import Namespace
from rdflib.namespace import RDF


RDF_EXTS = ['.owl', '.OWL', '.rdf', '.RDF', '.ttl', '.TTL']
GPML_EXTS = ['.gpml', '.GPML']
SBML_EXTS = ['.sbml', '.SBML']

BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")


# class for representing a pathway entity (borrows heavily from BioPAX)
class Entity:
    def __init__(self,
                 names: list,
                 identifiers: list,
                 enttype: str
                 ):
        self.names = names
        self.identifiers = identifiers
        self.enttype = enttype


# class for representing a pathway
class Pathway:
    def __init__(self,
                 pid: str,
                 names: list,
                 entities: list,
                 graph,
                 comments: list = [],
                 subpaths: list = []
                 ):
        self.pid = pid
        self.names = names
        self.entities = entities
        self.graph = graph
        self.comments = comments
        self.subpaths = subpaths

    def __repr__(self):
        return json.dumps(
            {
                'pid': self.pid,
                'names': self.names,
                'entities': self.entities
            }
        )


# class for representing a pathway KB
class PathKB:
    def __init__(self,
                 name: str
                 ):
        self.name = name
        self.pathways = []

    def load(self, location):
        """
        Sends file to appropriate reader, or iterate through files if given a directory
        :param location:
        :return:
        """
        if os.path.isfile(location):
            fname, fext = os.path.splitext(location)
            if fext in RDF_EXTS:
                return self.load_from_biopax(location)
            elif fext in GPML_EXTS:
                return self.load_from_gpml(location)
            elif fext in SBML_EXTS:
                return self.load_from_sbml(location)
            else:
                raise(NotImplementedError, "Unknown file type! {}".format(location))
        elif os.path.isdir(location):
            files = glob.glob(os.path.join(location, '*.*'))
            output = []
            for f in files:
                output += self.load(f)

    def load_from_biopax(self, loc):
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

        # initialize pathway list
        plist = []

        for pathway in list(g.subjects(RDF.type, BP3["Pathway"])):
            sys.stdout.write("%s\n" % pathway)

            # initialize pathway data components
            path_data = {}
            path_data["pid"] = pathway.encode('utf-8')
            path_data["names"] = list(g.objects(pathway, BP3['displayName']))[0].encode('utf-8') + \
                                 list(g.objects(pathway, BP3['standardName']))[0].encode('utf-8') + \
                                 list(g.objects(pathway, BP3['name']))[0].encode('utf-8')

            comments = list(g.objects(pathway, BP3["comment"]))
            path_data["comments"] = [c.encode('utf-8') for c in comments if not (c.startswith("Authored"))
                                                                        and not (c.startswith("Reviewed"))
                                                                        and not (c.startswith("Edited"))]

            subpaths = set([])

            graph = nx.Graph()

            for component in list(g.objects(pathway, BP3["pathwayComponent"])):
                # get type of component
                comp_type = str(list(g.objects(component, RDF.type))[0]).split('#')[-1]
                comp_name = str(component).split('#')[-1]

                # if component is a pathway
                if comp_type == "Pathway":
                    subpaths.add(str(component))
                # if component is in reaction set
                elif comp_type in use_rx_types:
                    left = list(g.objects(component, BP3["left"]))
                    right = list(g.objects(component, BP3["right"]))
                    product = list(g.objects(component, BP3["product"]))
                    participant = list(g.objects(component, BP3["participant"]))

                    participants += left + right + product + participant

                    for obj in left:
                        graph.add_edge(comp_name, str(obj).split('#')[-1], type="left")
                    for obj in right:
                        graph.add_edge(comp_name, str(obj).split('#')[-1], type="right")
                    for obj in product:
                        graph.add_edge(comp_name, str(obj).split('#')[-1], type="product")
                    for obj in participant:
                        graph.add_edge(comp_name, str(obj).split('#')[-1], type="participant")

                    control = g.subjects(BP3["controlled"], component)
                    for c in control:
                        controller = list(g.objects(c, BP3["controller"]))
                        participants += controller

                        for obj in controller:
                            graph.add_edge(comp_name, str(obj).split('#')[-1], type="controller")

            # gather all entities
            entSet, entRels = reduce_complexes_and_physEnts(g, set(participants), [], set([]))

            # add complex and physical entity relationship to graph
            for a, b, rel in entRels:
                graph.add_edge(str(a).split('#')[-1], str(b).split('#')[-1], type=rel)

            # set values for entity names, types, and xrefs
            for e in entSet:
                eid, enames, etype, exrefs = get_ref_ids(g, e, dnstr, bxref, bmem)
                entnames[eid] = enames
                enttypes[eid] = etype
                entxrefs[eid] = exrefs

            path_data["subpaths"] = subpaths
            path_data["entnames"] = entnames
            path_data["enttypes"] = enttypes
            path_data["entxrefs"] = entxrefs
            path_data["graph"] = graph

            plist.append(path_data)

        return plist

    def load_from_gpml(self, loc):
        """
        Loads pathway from GPML file
        :param loc: location of file
        :return:
        """
        return

    def load_from_sbml(self, loc):
        """
        Loads pathway from SBML
        :param loc: location of file
        :return:
        """
        raise(NotImplementedError, "SBML file reader not implemented yet!")





