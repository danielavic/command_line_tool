from os import mkdir
import uuid
from datetime import datetime
import pandas as pd
import networkx as nx
import matchmsextras.networking as net
from matchms.networking import SimilarityNetwork
from matchms.similarity import CosineGreedy
import os
from uuid import uuid4 
import numpy as np 

def to_pandas_edgelist(G, source="source",target="target",nodelist=None,dtype=None,order=None,edge_key=None):

    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)
    source_nodes = [s for s, _, _ in edgelist]
    target_nodes = [t for _, t, _ in edgelist]

    all_attrs = set().union(*(d.keys() for _, _, d in edgelist))
    if source in all_attrs:
        raise nx.NetworkXError(f"Source name {source!r} is an edge attr name")
    if target in all_attrs:
        raise nx.NetworkXError(f"Target name {target!r} is an edge attr name")

    nan = float("nan")
    edge_attr = {k: [d.get(k, nan) for _, _, d in edgelist] for k in all_attrs}

    if G.is_multigraph() and edge_key is not None:
        if edge_key in all_attrs:
            raise nx.NetworkXError(f"Edge key name {edge_key!r} is an edge attr name")
        edge_keys = [k for _, _, k in G.edges(keys=True)]
        edgelistdict = {source: source_nodes, target: target_nodes, edge_key: edge_keys}
    else:
        edgelistdict = {source: source_nodes, target: target_nodes}

    edgelistdict.update(edge_attr)
    return pd.DataFrame(edgelistdict, dtype=dtype)

def from_pandas_edgelist(df, source="source", target="target", edge_attr=None, create_using=None,edge_key=None):
    
    g = nx.empty_graph(0, create_using)

    if edge_attr is None:
        g.add_edges_from(zip(df[source], df[target]))
        return g

    reserved_columns = [source, target]

    # Additional columns requested
    attr_col_headings = []
    attribute_data = []
    if edge_attr is True:
        attr_col_headings = [c for c in df.columns if c not in reserved_columns]
    elif isinstance(edge_attr, (list, tuple)):
        attr_col_headings = edge_attr
    else:
        attr_col_headings = [edge_attr]
    if len(attr_col_headings) == 0:
        raise nx.NetworkXError(
            f"Invalid edge_attr argument: No columns found with name: {attr_col_headings}"
        )

    try:
        attribute_data = zip(*[df[col] for col in attr_col_headings])
    except (KeyError, TypeError) as err:
        msg = f"Invalid edge_attr argument: {edge_attr}"
        raise nx.NetworkXError(msg) from err

    if g.is_multigraph():
        # => append the edge keys from the df to the bundled data
        if edge_key is not None:
            try:
                multigraph_edge_keys = df[edge_key]
                attribute_data = zip(attribute_data, multigraph_edge_keys)
            except (KeyError, TypeError) as err:
                msg = f"Invalid edge_key argument: {edge_key}"
                raise nx.NetworkXError(msg) from err

        for s, t, attrs in zip(df[source], df[target], attribute_data):
            if edge_key is not None:
                attrs, multigraph_edge_key = attrs
                key = g.add_edge(s, t, key=multigraph_edge_key)
            else:
                key = g.add_edge(s, t)

            g[s][t][key].update(zip(attr_col_headings, attrs))
    else:
        for s, t, attrs in zip(df[source], df[target], attribute_data):
            g.add_edge(s, t)
            g[s][t].update(zip(attr_col_headings, attrs))

    return g


def individual_graphml_and_edge_list(scores, analysis, threshold):

    #graphml
    ms_network = SimilarityNetwork(identifier_key="feature_id", score_cutoff=threshold, keep_unconnected_nodes=False)
    
    ms_network.create_network(scores)
    our_network = ms_network.graph
    net = to_pandas_edgelist(our_network)
    net.rename(columns={'weight': analysis}, inplace=True)
    individual_graphml_and_edge_list.net = net

def min_matched(spectra):
    cosine_greedy = CosineGreedy(tolerance=0.2)
    tuples, score_array, min_matched, score, reference, query, ref, quer, reference1, query1 = ([] for i in range(10))
   
    for key, value in enumerate(spectra):
        ref.append(value)
        quer.append(value)
        
    for i in range(len(ref)):
        reference.append(ref[i].metadata['feature_id'])
        
    for i in range(len(quer)):
        query.append(quer[i].metadata['feature_id'])
        
    for i in range(len(ref)):
        for j in range(len(quer)):
            tuples.append([(reference[i],query[j])])
            #score_pair_test = cosine_greedy.pair(reference[i],query[j])
            score_array.append(cosine_greedy.pair(ref[i],quer[j]))
    
    for a in range(len(score_array)): 
        score.append(float(score_array[a]['score']))
        min_matched.append(int(score_array[a]['matches']))
    
    dict1 = {'ref/query': tuples , 'score': score, 'min_matched': min_matched }
    
    for a in range(len(dict1['ref/query'])):
        reference1.append(dict1['ref/query'][a][0][0])
        query1.append(dict1['ref/query'][a][0][1])
        
    dict2 = {'source': reference1, 'target': query1, 'cosine': score, 'matched_peaks': min_matched}
    
    df2 = pd.DataFrame.from_dict(dict2)
    
    for i in df2['matched_peaks']: 
        if i <= 6:
            df2.drop(df2[df2['matched_peaks'] <= 6].index, inplace = True)
        df2.drop(df2[df2['cosine'] < 0.7].index, inplace = True)
    
    
    return df2














