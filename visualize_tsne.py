import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

with open("../input/movielens-models/100k_True_3.pkl",'rb') as file:
    tmodel = pickle.load(file)
ug, mg, rg, ux, mx, rx, y = load_data100k(3,"base",True)
users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', sep="|", encoding='latin-1', header=None)
users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
genc = LabelEncoder()
users.gender = genc.fit_transform(users.gender)
zenc = LabelEncoder()
users["zip code"] = zenc.fit_transform(users["zip code"])
oenc = LabelEncoder()
users.occupation = oenc.fit_transform(users.occupation)


# fig = plt.figure(figsize = (25,5))

# for idx,separate in enumerate(users.columns):
#     ax = plt.subplot(1,5,idx+1)
#     outs = []
#     ty = []
#     for u in set(users[separate].values):
#         px = users.loc[users[separate]==u].values
#         out = tmodel.unet(ug,torch.tensor(px).cuda())#.cpu().detach().numpy()
#         outs.append(out)
#         ty.extend(np.ones(px.shape[0])*u)
#     tx = torch.cat(outs).cpu().detach().numpy()

#     txe = TSNE(n_components=2, n_jobs=-1).fit_transform(tx)
    
#     for idx, u in enumerate(set(users[separate].values)):
#         ux = ty==u
#         plt.scatter(txe[ux,0], txe[ux,1], label=u)
#     ax.set_title(separate)
# plt.savefig("../working/graph")


distance_matrix = pairwise_distances(xs,xs, metric='cosine', n_jobs=-1)
model = TSNE(metric="precomputed")
uxe = model.fit_transform(distance_matrix)
uxe = TSNE(n_components=2, n_jobs=-1, perplexity = 50).fit_transform(xs)

for u in set(outs.round()):
    idx = outs.round()==u
    plt.scatter(uxe[idx,0], uxe[idx,1])
    break