#External modules
from networkx import MultiGraph # type: ignore
from json import loads,dumps

################################################################################

"""
Printing and parsing functions

print_time
"""



def print_time(floatHours ):
	intHours = int(floatHours)
	return "%02d:%02d" % (intHours,(floatHours-intHours)*60)

def graph_to_json(G ):
    """
    Serialize a MultiGraph
    """
    from networkx import node_link_data
    return dumps(node_link_data(G))

def json_to_graph(raw):
    """
    Deserialize a MultiGraph
    """
    from networkx import node_link_graph
    return node_link_graph(loads(raw))
