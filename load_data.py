import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from dgl import DGLGraph
import numpy as np

genre = ['Action','Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',\
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',\
          'Thriller', 'War', 'Western']

def fc(nn, labels=None):
    a = list(range(nn))*nn
    b = sorted(list(range(nn)))*(nn-1)
    for i in range(nn):
        a.pop(nn*i)
    a,b = np.array(a),np.array(b)
    if labels is not None:
        a,b = np.array(labels)[a], np.array(labels)[b]
    return a,b

def load_data100k(fold,fformat,version="alpha"):
    reviews = pd.read_csv(f'../input/movielens-100k-dataset/ml-100k/u{fold}.{fformat}', sep="\t", header=None)
    reviews.columns = ['user id', 'movie id', 'rating', 'timestamp']

    info = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    info.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', *genre]
    
    users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', sep="|", encoding='latin-1', header=None)
    users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    
    enc = LabelEncoder()
    users.occupation = enc.fit_transform(users.occupation)
    users.gender = enc.fit_transform(users.gender)
    users["zip code"] = enc.fit_transform(users["zip code"])
        
    ml = pd.merge(pd.merge(info,reviews),users)
    movie_features = ["movie id", *genre] if version=="alpha" else genre
    ux = torch.tensor(ml[users.columns].values) if version=="alpha" else torch.tensor(ml[users.loc[:, users.columns != 'user id'].columns].values)
    mx = torch.tensor(ml[movie_features].values)
    y = torch.tensor(ml.rating.values)
    
    if version=="alpha":
        ug = DGLGraph().to(torch.device('cuda:0'))
        ug.add_nodes(5)
        ug.add_edges(*fc(5))

        mg = DGLGraph().to(torch.device('cuda:0'))
        mg.add_nodes(19)
        mg.add_edges(*fc(19))

        rg = DGLGraph().to(torch.device('cuda:0'))
        rg.add_nodes(2)
        rg.add_edges(*fc(2))
        
        return ug, mg, rg, ux, mx, torch.zeros(y.shape), y
        
    else:
        rx = torch.tensor(ml[["user id","movie id"]].values)
        
        ug = DGLGraph().to(torch.device('cuda:0'))
        ug.add_nodes(4)
        ug.add_edges(*fc(4))

        mg = DGLGraph().to(torch.device('cuda:0'))
        mg.add_nodes(18)
        mg.add_edges(*fc(18))

        rg = DGLGraph().to(torch.device('cuda:0'))
        rg.add_nodes(2)
        rg.add_edges(*fc(2))

        return ug, mg, rg, ux, mx, rx, y

def load_data1M(version='alpha'):
    reviews = pd.read_csv('../input/movielens-1m/ml-1m/ratings.dat', delimiter='::', engine='python', header = None)
    reviews.columns = ['user id', 'movie id', 'rating', 'timestamp']

    info = pd.read_csv('../input/movielens-1m/ml-1m/movies.dat', delimiter='::', engine='python', header = None)
    info.columns = ['movie id', 'movie title', 'genre']
    s = info.genre.str.split('|',expand=True)
    x = pd.DataFrame.from_dict({k:np.zeros(len(s)) for k in genre})
    for i in s.columns:
        d = pd.get_dummies(s[i])
        x[d.columns] += d
    info.drop("genre",1)
    info[genre] = x

    users = pd.read_csv('../input/movielens-1m/ml-1m/users.dat', delimiter='::', engine='python', header = None)
    users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

    enc = LabelEncoder()
    users.occupation = enc.fit_transform(users.occupation)
    users.gender = enc.fit_transform(users.gender)
    users["zip code"] = enc.fit_transform(users["zip code"])

    ml = pd.merge(pd.merge(info,reviews),users)
    movie_features = ["movie id", *genre] if version=="alpha" else genre
    ux = torch.tensor(ml[users.columns].values) if version=="alpha" else torch.tensor(ml[users.loc[:, users.columns != 'user id'].columns].values)
    mx = torch.tensor(ml[movie_features].values)
    y = torch.tensor(ml.rating.values)
    
    if version=="alpha":
        ug = DGLGraph().to(torch.device('cuda:0'))
        ug.add_nodes(5)
        ug.add_edges(*fc(5))

        mg = DGLGraph().to(torch.device('cuda:0'))
        mg.add_nodes(19)
        mg.add_edges(*fc(19))

        rg = DGLGraph().to(torch.device('cuda:0'))
        rg.add_nodes(2)
        rg.add_edges(*fc(2))
        
        return ug, mg, rg, ux, mx, torch.zeros(y.shape), y
        
    else:
        rx = torch.tensor(ml[["user id","movie id"]].values)
        
        ug = DGLGraph().to(torch.device('cuda:0'))
        ug.add_nodes(4)
        ug.add_edges(*fc(4))

        mg = DGLGraph().to(torch.device('cuda:0'))
        mg.add_nodes(18)
        mg.add_edges(*fc(18))

        rg = DGLGraph().to(torch.device('cuda:0'))
        rg.add_nodes(2)
        rg.add_edges(*fc(2))

        return ug, mg, rg, ux, mx, rx, y
    