import numpy as np
qf45 = np.load('qf_ep45_exp15.npy')
gf45 = np.load('gf_ep45_exp15.npy')
qf55 = np.load('qf_ep55_exp15.npy')
gf55 = np.load('gf_ep55_exp15.npy')
qf_ens = qf45 + qf55
qf_ens = qf_ens / np.linalg.norm(qf_ens, axis=1, keepdims=True)
gf_ens = gf45 + gf55
gf_ens = gf_ens / np.linalg.norm(gf_ens, axis=1, keepdims=True)
np.save('qf_ensemble_exp15_45_55.npy', qf_ens)
np.save('gf_ensemble_exp15_45_55.npy', gf_ens)
print('Done. qf:', qf_ens.shape, 'gf:', gf_ens.shape)