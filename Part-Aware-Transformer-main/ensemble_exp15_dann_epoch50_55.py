import numpy as np
qf50 = np.load('qf_ep50_exp15.npy')
gf50 = np.load('gf_ep50_exp15.npy')
qf55 = np.load('qf_ep55_exp15.npy')
gf55 = np.load('gf_ep55_exp15.npy')
qf_ens = qf50 + qf55
qf_ens = qf_ens / np.linalg.norm(qf_ens, axis=1, keepdims=True)
gf_ens = gf50 + gf55
gf_ens = gf_ens / np.linalg.norm(gf_ens, axis=1, keepdims=True)
np.save('qf_ensemble_exp15_50_55.npy', qf_ens)
np.save('gf_ensemble_exp15_50_55.npy', gf_ens)
print('Done. qf:', qf_ens.shape, 'gf:', gf_ens.shape)