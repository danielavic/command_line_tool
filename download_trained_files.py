import requests

def download_trained_models(): 

    url1 = 'https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model?download=1'
    url2 = 'https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.trainables.syn1neg.npy?download=1'
    url3 = 'https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.wv.vectors.npy?download=1'

    response1 = requests.get(url1)
    response2 = requests.get(url2)
    response3 = requests.get(url3)
    open("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model", "wb").write(response1.content)
    open("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.trainables.syn1neg.npy", "wb").write(response2.content)
    open("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.wv.vectors.npy", "wb").write(response3.content)
