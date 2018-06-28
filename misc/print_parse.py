#External modules
from networkx import node_link_graph,node_link_data,MultiGraph # type: ignore
from json import loads,dumps

################################################################################

"""
Printing and parsing functions

print_time
"""



def print_time(floatHours : float) -> str:
	intHours = int(floatHours)
	return "%02d:%02d" % (intHours,(floatHours-intHours)*60)

def graph_to_json(G : MultiGraph)->str:
    """
    Serialize a MultiGraph
    """
    return dumps(node_link_data(G))

def json_to_graph(raw : str) -> MultiGraph:
    """
    Deserialize a MultiGraph
    """
    return node_link_graph(loads(raw))
