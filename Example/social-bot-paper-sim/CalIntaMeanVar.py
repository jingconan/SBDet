from SBDet import *
import pylab as P
tr = zload('com-det-compare-res.pkz')
solution_leading_eigen = tr['solution_leading_eigen']
solution_info_map = tr['solution_info_map']
sol_ref = tr['ref_sol']
solution_sdp = tr['solution']
inta = tr['inta']
inta = P.array(inta)


sol_tp = ['ref', 'our', 'newman', 'infomap']
sol_names = ['ref_sol', 'solution', 'solution_leading_eigen', 'solution_info_map']
print('method\tnormal mean\tnormal std\tbot mean\tbot std')
for i, name in enumerate(sol_names):
    sol = tr[name]
    sol = np.array(sol)
    n_m = P.mean(inta[sol<0.5])
    n_v = P.std(inta[sol<0.5])
    b_m = P.mean(inta[sol>0.5])
    b_v = P.std(inta[sol>0.5])
    print('%s\t%f\t%f\t%f\t%f' %(sol_tp[i], n_m, n_v, b_m, b_v))

