import click
from datetime import datetime
from pyteomics import mgf
import numpy as np
from matchms.importing import load_from_mgf
import io 
import requests 
import os 
from uuid import uuid4
from graphml_and_edge_table import individual_graphml_and_edge_list, min_matched, from_pandas_edgelist, to_pandas_edgelist
#from data_processing import metadata_processing, peak_processing, spectrum_processing
from data_processing import spectrum_processing
import gensim
from download_trained_files import download_trained_models
from spec2vec import Spec2Vec
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine
import networkx as nx
import matchmsextras.networking as net
from matchms.networking import SimilarityNetwork
from graphml_and_edge_table import individual_graphml_and_edge_list
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles

@click.command()
@click.option("--taskid",
                  help="nothing", type=str)
@click.option("--trained_file",
                  help="nothing", type=str)
@click.option("--threshold_spec2vec",
                  help="nothing", type=float)
@click.option("--threshold_cosine",
                  help="nothing", type=float)
@click.option("--min_matches",
                  help="nothing", type=str)

def spectrum_similarity_tool (taskid, trained_file,threshold_spec2vec, threshold_cosine, min_matches):
    #Generate and load spectra
    current_datetime = datetime.now()
    filename = f"{current_datetime}--{taskid}.mgf"
    path = f"mgf_files/{filename}"
    #path = 'mgf_files/test2.mgf'
    url_to_spectra = "http://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile?task=%s&block=main&file=spec/" % taskid
    spectra = []
    with mgf.MGF(io.StringIO(requests.get(url_to_spectra).text)) as reader:
        for spectrum in reader:
            spectra.append(spectrum)

    #save spectra
    mgf.write(spectra, output= path, key_order = ['feature_id', 'pepmass', 'scans', 'charge', 
                                                                    'mslevel', 'retention_time', 'precursor_mz',
                                                                    'ionmode', 'inchi', 'inchikey', 'smiles' ])
    
    spectra = list(load_from_mgf(path))

    spectra = [spectrum_processing(s) for s in load_from_mgf(path)]
    spectra = [s for s in spectra if s is not None]

    if trained_file == 'yes':
        model = gensim.models.Word2Vec.load("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    elif trained_file == 'no': 
        download_trained_models()
        path_model = os.path.join(os.path.dirname(os.getcwd()),
                    "data", "trained_models") 
                           # enter your path name when different
        filename_model = "downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
        filename = os.path.join(path_model, filename_model)
        model = gensim.models.Word2Vec.load("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    else: 
        print('Please, set the trained_file parameter to no')

    spec2vec_similarity = Spec2Vec(model=model,
                               intensity_weighting_power=0.5,
                               allowed_missing_percentage=5.0)

    scores_spec2vec = calculate_scores(spectra, spectra, spec2vec_similarity,
                          is_symmetric=True)
 
    individual_graphml_and_edge_list(scores_spec2vec, 'spec2vec', threshold_spec2vec)
    
    spec2vec_net = individual_graphml_and_edge_list.net
    print('calculei spec2vec')
 

    # Calculate cosine similarity scores
    '''scores_cosine = calculate_scores(spectra, spectra, CosineGreedy())

    individual_graphml_and_edge_list(scores_cosine, 'cosine', threshold_cosine)
    
    
    cosine_net = individual_graphml_and_edge_list.net'''
   
    if min_matches == 'yes':
        cosine_net = min_matched(spectra)
        index_names = cosine_net[ (cosine_net['source'] == cosine_net['target'])].index
        cosine_net.drop(index_names, inplace = True)
        
    else:
        scores_cosine = calculate_scores(spectra, spectra, CosineGreedy())
        individual_graphml_and_edge_list(scores_cosine, 'cosine', threshold_cosine)
        cosine_net = individual_graphml_and_edge_list.net

    # Merge tables 

    spec2vec_net['source']=spec2vec_net['source'].astype(int)
    spec2vec_net['target']=spec2vec_net['target'].astype(int)
    cosine_net['source']= cosine_net['source'].astype(int)
    cosine_net['target']= cosine_net['target'].astype(int)

    merged = pd.merge(cosine_net, spec2vec_net, how='outer', on=["source", "target"], indicator=True)
    merged_zeros = merged[merged.columns[:-1]].fillna(0)
    merged_zeros['_merge'] = merged['_merge']

    df = merged_zeros
    #graphml_filename = f"graphml_files/{current_datetime}--{outputname}--{uuid4()}.graphml"
    graphml_filename = f"graphml_files/{current_datetime}--{taskid}--{uuid4()}.graphml"
    #session["graphml_filename"]=graphml_filename
    G = from_pandas_edgelist(df, source='source', target='target', edge_attr=True)
    nx.write_graphml(G, graphml_filename)
    H = nx.read_graphml(graphml_filename)
    edgelist = to_pandas_edgelist(H)
    #filename_csv_file = f"csv_files/{current_datetime}--edge_table_merged--{taskid}--{uuid4()}.csv"
    filename_csv_file = f"csv_files/{current_datetime}--edge_table_merged--{taskid}--{uuid4()}.csv"
    edgelist.to_csv(filename_csv_file)
    print('juntei tabelas!')

    ## Venn Diagram

    df = pd.read_csv(filename_csv_file)
    
    venn2(subsets= (df._merge.value_counts().left_only, df._merge.value_counts().right_only, df._merge.value_counts().both), 
                    set_labels= ('Cosine', 'Spec2vec'))
    
    #plt.savefig(f"images/{current_datetime}--venndiagram--{uuid4()}.png")
    plt.savefig(f"images/{current_datetime}--venndiagram--{uuid4()}.png")
    
    # Thresholds
    print(f'For {threshold_spec2vec} as spec2vec threshold and {threshold_cosine} as cosine threshold:')
    # Number of nodes 
    nodes = G.number_of_nodes()
    print(f'The number of nodes in the network:{nodes}')
    # Number of edges
    edges = G.number_of_edges()
    print(f'The number of edges in the network: {edges}')
    # Edges intersection
    intersection = df._merge.value_counts().both
    print(f'The number of edges found for both cosine and spec2vec: {intersection}')
    # Cosine 
    edges_cosine = df._merge.value_counts().left_only + df._merge.value_counts().both
    print(f'The number of edges for cosine: {edges_cosine}')
    #Spec2vec
    edges_spec = df._merge.value_counts().right_only + df._merge.value_counts().both
    print(f'The number of edges for spec2vec: {edges_spec}')


if __name__ == "__main__":
    spectrum_similarity_tool()