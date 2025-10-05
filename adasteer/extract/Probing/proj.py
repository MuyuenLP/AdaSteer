from utils import load_from_pickle, save_to_pickle
from tqdm import tqdm
from probe import get_orthogonalized_matrix, get_orthogonalized_matrix_2
import numpy as np

r = load_from_pickle("vectors/gemma2-9b-it/RD/refusal.pkl")
w = load_from_pickle("vectors/gemma2-9b-it/HD/refusal.pkl")

w_Orthogonalization = []
proj = []
for l in tqdm(range(32)):
    r_l = r[l]
    w_l = w[l]

    proj_l = get_orthogonalized_matrix_2(w_l, r_l, oth_scale).cpu().numpy()

    proj.append(proj_l)

proj = np.stack(proj)
# print(w_Orthogonalization.shape)

print(proj)

save_to_pickle(proj, "vectors/gemma2-9b-it/HD/proj.pkl")