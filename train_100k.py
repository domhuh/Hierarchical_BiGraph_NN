from .load_data import *
from .networks import *
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math

def evaluate(net, vdl):
    eloss = 0
    net.eval()
    with torch.no_grad():
        for i,(u,m,r,label) in enumerate(vdl):
            out = net(ug, mg, rg, u.cuda(), m.cuda(), r.cuda())
            loss = torch.sqrt(f.mse_loss(out, label.cuda().float()))
            eloss += loss.item()
    return eloss/i

torch.cuda.set_device(0)

place_dict = {}
link_dict = {}

for link_nr in range(1,11):
    for place_nr in range(1,11):
        for attention in [True,False]:
            for use_id in [True,False]:
                for FOLD in range(1,5+1):
                    ug, mg, rg, ux, mx, rx, y = load_data100k(FOLD,"base",use_id)
                    tdl = DataLoader(TensorDataset(ux,mx,rx,y), batch_size=128, shuffle=True)
                    ug, mg, rg, ux, mx, rx, y = load_data100k(FOLD,"test",use_id)
                    vdl = DataLoader(TensorDataset(ux,mx,rx,y), batch_size=128, shuffle=True)

                    user =  getEmbeddingSize(ux)
                    if use_id: #set age to fixed 100 vocab
                        user[1] = 100
                    else: user[0] = 100

                    embedding_size = [getEmbeddingSize(mx), user, getEmbeddingSize(rx)]


                    net = RatingNet(512,512,1024,4028,1, use_id, embedding_size, attention, num_rounds=[link_nr,place_nr]).cuda()
                    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, amsgrad=True)

                    pb = tqdm(range(10))

                    for epoch in pb:
                        net.train()
                        eloss = 0
                        for i,(u,m,r,label) in enumerate(tdl):
                            out = net(ug, mg, rg, u.cuda(), m.cuda(), r.cuda())
                            loss = f.mse_loss(out, label.cuda().float())
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            eloss += math.sqrt(loss.detach().item())
                            pb.set_description(f"{epoch}| {eloss/(i+1)} | {i}")

                        vloss = evaluate(net, vdl)
                        eloss /= i

                        net.tl.append(eloss)
                        net.vl.append(vloss)
                    
                    place_dict[place_nr] = [net.tl, net.vl]
                    link_dict[link_nr] = [net.tl, net.vl]
                    
                    with open(f"../working/100k_{use_id}_{FOLD}.pkl", 'wb') as file:
                        pickle.dump(net,file)

                    with open(f"../working/100k_{use_id}_{FOLD}_p.pkl", 'wb') as file:
                        pickle.dump(place_dict,file)
                    with open(f"../working/100k_{use_id}_{FOLD}_l.pkl", 'wb') as file:
                        pickle.dump(link_dict,file)
