import torch.nn as nn
import torch.nn.functional as f
import dgl.function as fn
import dgl

def gcn_msg(edges):
    return {'m': edges.src['h']}

def gcn_attentive_reduce(nodes): 
    return {'h': nodes.data['h'], 'ms': nodes.mailbox['m']}

def gcn_reduce(nodes):
    return {'h': nodes.data['h'], 'ms': torch.sum(nodes.mailbox['m'],1)}

def create_graph(g, feature):
    ng = dgl.DGLGraph()
    n_node = g.number_of_nodes()
    ng.add_nodes(n_node)
    ng.add_edges(*fc(n_node))
    ng.ndata['h'] = feature
    return ng

class UserLayer(nn.Module):
    def __init__(self, ni, no, use_id = True, embedding_sizes=None, attention=False, num_rounds=1):
        super(UserLayer, self).__init__()
        self.use_id = use_id
        self.n_nodes = 5 if self.use_id else 4
        self.fc = nn.Linear(ni*self.n_nodes, no)
        self.num_rounds = num_rounds
        
        self.attention = attention
        if self.attention:
            self.key = nn.Linear(ni, ni)
            self.query = nn.ModuleList()
            for _ in range(self.n_nodes):
                self.query.append(nn.Linear(ni, ni))
#       self.value = nn.Linear(ni,ni)
        
        self.node = nn.GRUCell(ni*self.n_nodes, ni*self.n_nodes)
        if self.use_id:
            self.id = nn.Embedding(embedding_sizes[0],ni)
            self.age = nn.Embedding(embedding_sizes[1],ni)
            self.gender = nn.Embedding(embedding_sizes[2],ni)
            self.occupation = nn.Embedding(embedding_sizes[3],ni)
            self.zipcode = nn.Embedding(embedding_sizes[4],ni)
        else:
            self.age = nn.Embedding(embedding_sizes[0],ni)
            self.gender = nn.Embedding(embedding_sizes[1],ni)
            self.occupation = nn.Embedding(embedding_sizes[2],ni)
            self.zipcode = nn.Embedding(embedding_sizes[3],ni)
        
    def forward(self, g, feature):
        bs = feature.shape[0]
        with g.local_scope():
            features = self.embed(feature.long())
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(self.num_rounds):
                if self.attention:
                    g.update_all(gcn_msg, gcn_attentive_reduce)
                    h = g.ndata['h']
                    q = self.apply_query(g.ndata['ms'])
                    atfm = torch.softmax(torch.bmm(q,self.key(h).unsqueeze(1).transpose(1,-1)),1)
                    x = atfm*g.ndata['ms']
                    x = torch.sum(x,dim=1).reshape(bs,-1)
                    h = h.reshape(bs,-1)
                else:
                    g.update_all(gcn_msg, gcn_reduce)
                    x = g.ndata['ms'].reshape(bs,-1)
                    h = g.ndata['h'].reshape(bs,-1)
                g.ndata['h'] = self.node(x,h).reshape(bs*self.n_nodes,-1) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)
    
    def embed(self, feature):
        if self.use_id:
            id = self.id(feature[:,0].unsqueeze(1))#f.one_hot(feature[:,0])
            age = self.age(feature[:,1].unsqueeze(1))#f.one_hot(feature[:,0])
            gender = self.gender(feature[:,2].unsqueeze(1))#f.one_hot(feature[:,1])
            occupation = self.occupation(feature[:,3].unsqueeze(1))#f.one_hot(feature[:,2])
            zipcode = self.zipcode(feature[:,4].unsqueeze(1))#f.one_hot(feature[:,2])
            return torch.cat([id,age,gender,occupation,zipcode],1)
        else:
            age = self.age(feature[:,0].unsqueeze(1))#f.one_hot(feature[:,0])
            gender = self.gender(feature[:,1].unsqueeze(1))#f.one_hot(feature[:,1])
            occupation = self.occupation(feature[:,2].unsqueeze(1))#f.one_hot(feature[:,2])
            zipcode = self.zipcode(feature[:,3].unsqueeze(1))#f.one_hot(feature[:,2])
            return torch.cat([age,gender,occupation,zipcode],1)
    
    def apply_query(self,x):
        return torch.cat([self.query[i](x[:,i]).unsqueeze(1) for i in range(self.n_nodes-1)],1)
    
class MovieLayer(nn.Module):
    def __init__(self, ni, no, use_id = True, embedding_sizes=None, attention=False, num_rounds = 1):
        super(MovieLayer, self).__init__()
        self.use_id = use_id
        self.n_nodes = 19 if self.use_id else 18
        self.fc = nn.Linear(ni*self.n_nodes, no)
        self.hidden = nn.Linear(ni,ni)
        self.update = nn.Linear(ni,ni)
        self.node = nn.GRUCell(ni*self.n_nodes, ni*self.n_nodes)
        self.num_rounds = num_rounds
        
        self.embedding = nn.ModuleList()
        for i in range(self.n_nodes):
            self.embedding.append(nn.Embedding(embedding_sizes[i],ni))
            
        self.attention = attention
        if self.attention:
            self.key = nn.Linear(ni, ni)
            self.query = nn.ModuleList()
            for _ in range(self.n_nodes):
                self.query.append(nn.Linear(ni, ni))

        
    def forward(self, g, feature):
        bs = feature.shape[0]
        with g.local_scope():
            features = self.embed(feature.long())
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(self.num_rounds):
                if self.attention:
                    g.update_all(gcn_msg, gcn_attentive_reduce)
                    h = g.ndata['h']
                    q = self.apply_query(g.ndata['ms'])
                    atfm = torch.softmax(torch.bmm(q,self.key(h).unsqueeze(1).transpose(1,-1)),1)
                    x = atfm*g.ndata['ms']
                    x = torch.sum(x,dim=1).reshape(bs,-1)
                    h = h.reshape(bs,-1)
                else:
                    g.update_all(gcn_msg, gcn_reduce)
                    x = g.ndata['ms'].reshape(bs,-1)
                    h = g.ndata['h'].reshape(bs,-1)
                #g.ndata['h'] = torch.tanh(self.hidden(g.ndata['h'])+self.update(g.ndata['ms'])) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)
    
    def embed(self, feature):
        return torch.cat([self.embedding[i](feature[:,i].unsqueeze(1)) for i in range(self.n_nodes)],1)
    
    def apply_query(self,x):
        return torch.cat([self.query[i](x[:,i]).unsqueeze(1) for i in range(self.n_nodes-1)],1)
    
class UserNet(nn.Module):
    def __init__(self, ni, nf, no, use_id=True, embedding_size=None,attention=False, num_rounds=1):
        super(UserNet, self).__init__()
        self.layer1 = UserLayer(ni, nf, use_id, embedding_size, attention, num_rounds)
        self.layer2 = nn.Linear(nf, no)
        
    def forward(self, g, features):
        x = torch.relu(self.layer1(g, features))
        x = self.layer2(x)
        return x

class MovieNet(nn.Module):
    def __init__(self, ni, nf, no, use_id=True, embedding_size=None,attention=False, num_rounds=1):
        super(MovieNet, self).__init__()
        self.layer1 = MovieLayer(ni, nf, use_id, embedding_size,attention, num_rounds)
        self.layer2 = nn.Linear(nf, no)
        
    def forward(self, g, features):
        x = torch.relu(self.layer1(g, features))
        x = self.layer2(x)
        return x

class RatingLayer(nn.Module):
    def __init__(self, ni, no, use_id=True, embedding_size = None, num_rounds=1):
        super(RatingLayer, self).__init__()
        self.use_id = use_id
        self.fc = nn.Linear(ni*2, no)
        self.node = nn.GRUCell(ni*2, ni*2)
        self.num_rounds = num_rounds
        if not self.use_id:
            self.rxu = nn.Embedding(embedding_size[0], ni)
            self.mergeu = nn.Linear(ni*2, ni)
            self.rxm = nn.Embedding(embedding_size[1], ni)
            self.mergem = nn.Linear(ni*2, ni)

    def forward(self, g, features, rx = None):
        if not self.use_id:
            up = self.mergeu(torch.cat([self.rxu(rx[:,0]),features[:,0]],1))
            mp = self.mergem(torch.cat([self.rxm(rx[:,1]),features[:,1]],1))
            feature = torch.cat([up,mp],1)
        bs = features.shape[0]
        with g.local_scope():
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(self.num_rounds):
                g.update_all(gcn_msg, gcn_reduce)
                x = g.ndata['ms'].reshape(bs,-1)
                h = g.ndata['h'].reshape(bs,-1)
                g.ndata['h'] = self.node(x,h).reshape(bs*2,-1)
            #g.ndata['h'] = torch.tanh(self.hidden(g.ndata['h'])+self.update(g.ndata['ms'])) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)

        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        
class RatingNet(nn.Module):
    def __init__(self, ni, nf, nir, nfr, no, use_id=True, embedding_size=None, attention=False, num_rounds=[1,1]):
        super(RatingNet, self).__init__()
        self.profile_es = ni
        self.rating_es = nir
        self.mnet = MovieNet(ni, nf, nir, use_id, embedding_size[0], attention, num_rounds[0])
        self.unet = UserNet(ni, nf, nir, use_id, embedding_size[1], attention, num_rounds[0])
        if use_id:
            self.layer1 = RatingLayer(nir, nfr, use_id, num_rounds=num_rounds[1])
        else:
            self.layer1 = RatingLayer(nir, nfr, use_id, embedding_size[2], num_rounds=num_rounds[1])
        self.layer2 = MLP(get_layers(nfr,512,no,128))
        self.apply(init_weights)
        
        self.tl = []
        self.vl = []
    def forward(self, ug, mg, rg, ufeatures, mfeatures, rfeatures=None):
        uxt = self.unet(ug, ufeatures)
        mxt = self.mnet(mg, mfeatures)
        rxt = torch.cat([mxt.unsqueeze(1),uxt.unsqueeze(1)],1)
        x = torch.relu(self.layer1(rg,rxt,rx=rfeatures))
        x = self.layer2(x)
        return x.squeeze()
    
def get_layers(start, hs, end, step):
    lse = [*list(range(hs, end, -step)), end]
    return list(zip([start,*lse[:]], [*lse[:], end]))[:-1]

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.LeakyReLU()) for n in layers])
    def forward(self, x):
        return self.model(x)

class umMLP(nn.Module):
    def __init__(self, layers, ne, embedding_size= None, partitions = []):
        super().__init__()
        self.partitions = partitions
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.LeakyReLU()) for n in layers])
        self.embedding = nn.ModuleList()
        for i in embedding_size:
            self.embedding.append(nn.Embedding(i,ne))
        self.apply(init_weights)
    def forward(self, ux, mx):
        u = torch.cat([self.embedding[i](ux[:,i-self.partitions[0]].unsqueeze(1)) for i in range(self.partitions[0], self.partitions[0]+self.partitions[1])],1)
        m = torch.cat([self.embedding[i](mx[:,i].unsqueeze(1)) for i in range(self.partitions[0])],1)
        x = torch.cat([u,m],1).flatten(1)
        return self.model(x)

def getEmbeddingSize(x):
    if (x==0).sum() == x.shape[0]:
        return []
    return (x.max(0).values + 1).numpy().astype(np.int)