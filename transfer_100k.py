from .load_data import *
from .networks import *
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math
import torch.nn as nn
import torch

def evaluate(net, vdl):
    eloss = 0
    net.eval()
    with torch.no_grad():
        for i,(u,m,r,label) in enumerate(vdl):
            out = net(ug, mg, rg, u.cuda(), m.cuda(), r.cuda())
            loss = torch.sqrt(f.mse_loss(out, label.cuda().float()))
            eloss += loss.item()
    return eloss/i

for version in ["alpha", "beta"]:
    ug, mg, rg, ux, mx, rx, y = load_data100k(version)
    tdl = DataLoader(TensorDataset(ux,mx,rx,y), batch_size=512, shuffle=True)
    ug, mg, rg, ux, mx, rx, y = load_data100k(version)
    vdl = DataLoader(TensorDataset(ux,mx,rx,y), batch_size=512, shuffle=True)

    embedding_size = [getEmbeddingSize(mx), getEmbeddingSize(ux), getEmbeddingSize(rx)]

    #alpha

    with open("../working/1M_alpha_3.pkl",'rb') as file:
        alpha_model = pickle.load(file)
        
    #reset embeddings
    alpha_model.mnet.layer1.embedding[0] = nn.Embedding(embedding_size[0][0], alpha_model.rating_es)
    alpha_model.unet.layer1.id = nn.Embedding(embedding_size[1][0], alpha_model.rating_es)
    alpha_model.unet.layer1.zipcode = nn.Embedding(embedding_size[1][-1], alpha_model.profile_es)

    alpha_optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=1e-3, amsgrad=True)

    for epoch in pb:
        alpha_model.train()
        eloss = 0
        for i,(u,m,r,label) in enumerate(tdl):
            out = alpha_model(ug, mg, rg, u.cuda(), m.cuda(), r.cuda())
            loss = f.mse_loss(out, label.cuda().float())
            alpha_optimizer.zero_grad()
            loss.backward()
            alpha_optimizer.step()
            eloss += math.sqrt(loss.detach().item())
            pb.set_description(f"{epoch}| {eloss/(i+1)} | {i}")

        vloss = evaluate(alpha_model, vdl)
        eloss /= i

        alpha_model.tl.append(eloss)
        alpha_model.vl.append(vloss)

    with open(f"../working/100k_alpha_pretrained_v{version}.pkl", 'wb') as file:
        pickle.dump(alpha_model,file)

    #beta

    with open("../working/1M_beta_3.pkl",'rb') as file:
        beta_model = pickle.load(file)  

    #reset embeddings
    beta_model.layer1.rxu = nn.Embedding(embedding_size[2][0], beta_model.rating_es).cuda()
    beta_model.layer1.rxm = nn.Embedding(embedding_size[2][1], beta_model.rating_es).cuda()
    beta_model.unet.layer1.zipcode = nn.Embedding(embedding_size[1][-1], beta_model.profile_es).cuda()
    
    beta_optimizer = torch.optim.AdamW(beta_model.parameters(), lr=1e-3, amsgrad=True)

    pb = tqdm(range(1))

    for epoch in pb:
        beta_model.train()
        eloss = 0
        for i,(u,m,r,label) in enumerate(tdl):
            out = beta_model(ug, mg, rg, u.cuda(), m.cuda(), r.cuda())
            loss = f.mse_loss(out, label.cuda().float())
            beta_optimizer.zero_grad()
            loss.backward()
            beta_optimizer.step()
            eloss += math.sqrt(loss.detach().item())
            pb.set_description(f"{epoch}| {eloss/(i+1)} | {i}")

        vloss = evaluate(beta_model, vdl)
        eloss /= i

        beta_model.tl.append(eloss)
        beta_model.vl.append(vloss)

    with open(f"../working/100k_beta_pretrained_v{version}.pkl", 'wb') as file:
        pickle.dump(beta_model,file)