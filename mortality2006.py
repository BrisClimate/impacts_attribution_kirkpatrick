from __future__ import division
import glob
import math
import numpy as np
import scipy.stats as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from datetime import datetime


def return_time_plot(plt_obj, ens, direction="ascending", period=1):
	# first calculate the return time data
	y_data, x_data = calc_return_times(ens, direction, period)
	l1 = plt_obj.semilogx(x_data, y_data, 'ko', marker='o', mec='k', mfc='w')
	return l1

def calc_return_times(em, direction, period):
    ''' calculate return times '''

    ey_data = em.flatten()
    ey_data.sort()
    	# reverse if necessary
    if direction == "descending":	# being beneath a threshold value
       ey_data = ey_data[::-1]
       
    	# create the n_ens / rank_data
    val = float(len(ey_data)) * 1.0/period
    end = float(len(ey_data)) * 1.0/period
    start = 1.0
    step = (end - start) / (len(ey_data)-1)
    ranks = [x*step+start for x in range(0, len(ey_data))]
    ex_data = val / np.array(ranks, dtype=np.float32)
    #print val,end,ranks
    return ey_data, ex_data


def calc_return_time_confidences(em, direction="descending", c=[0.05, 0.95], bsn=1000):
#def calc_return_time_confidences(em, direction, c, bsn):
	# c = confidence intervals (percentiles) to calculate
	# bsn = boot strap number, number of times to resample the distribution
	ey_data = em.flatten()
	# create the store
	sample_store = np.zeros((bsn, ey_data.shape[0]), 'f')
	# do the resampling
	for s in range(0, int(bsn)):
		t_data = np.zeros((ey_data.shape[0]), 'f')
		for y in range(0, ey_data.shape[0]):
                 x = np.int(np.random.uniform(0, ey_data.shape[0]))

                 t_data[y] = ey_data[x]
		t_data.sort()
		# reverse if necessary
		if direction == "descending":
			t_data = t_data[::-1]
		sample_store[s] = t_data
	# now for each confidence interval find the  value at the percentile
	conf_inter = np.zeros((len(c), ey_data.shape[0]), 'f')
	for c0 in range(0, len(c)):
		for y in range(0, ey_data.shape[0]):
			data_slice = sample_store[:,y]
			conf_inter[c0,y] = sci.scoreatpercentile(data_slice, c[c0]*100)
	return conf_inter


path = "../Mortality_data/"
tpath = "/home/bridge/yl17544/mortality_attr_results/tmean/"

all_hist = np.genfromtxt(path+"lndn_2006-06_2006-07_heatattributabledeaths_CAM5-1-1degree_All-Hist_100members.txt", dtype=float, skip_header=1, unpack=True)

nat_hist = np.genfromtxt(path+"lndn_2006-06_2006-07_heatattributabledeaths_CAM5-1-1degree_Nat-Hist_100members.txt", dtype=float, skip_header=1, unpack=True)

obs = np.genfromtxt(path+"lndn_2006-06_2006-07_heatattributabledeaths_MCC_obs.txt", dtype=float, skip_header=1, unpack=True)



all_hist_t = np.genfromtxt(tpath+"lndn_2006-06_2006-07_dailytemperatures_CAM5-1-1degree_All-Hist_100members.txt", dtype=float, skip_header=1, unpack=True)

nat_hist_t = np.genfromtxt(tpath+"lndn_2006-06_2006-07_dailytemperatures_CAM5-1-1degree_Nat-Hist_100members.txt", dtype=float, skip_header=1, unpack=True)

obs_t = np.genfromtxt(tpath+"lndn_2006-06_2006-07_dailytemperatures_MCC_obs.txt", dtype=float, skip_header=1, unpack=True)

tstart = 0

all_data = np.max(all_hist[:,tstart:],axis=1)
nat_data = np.max(nat_hist[:,tstart:],axis=1)
obs_data = np.max(obs[tstart:])

all_data_t = np.max(all_hist_t[:,tstart:],axis=1)
nat_data_t = np.max(nat_hist_t[:,tstart:],axis=1)
obs_data_t = np.max(obs_t[tstart:])

bsn = 10000

all_rt = calc_return_time_confidences(all_data,bsn=bsn)
nat_rt = calc_return_time_confidences(nat_data,bsn=bsn)

all_rt_t = calc_return_time_confidences(all_data_t,bsn=bsn)
nat_rt_t = calc_return_time_confidences(nat_data_t,bsn=bsn)

y_data, x_data = calc_return_times(all_data, direction='descending',period=1)

plt.vlines(x=10,ymin=-5,ymax=105,linestyle='dashed',alpha=0.5,color='lightgray')
#plt.vlines(x=50,ymin=-5,ymax=105,linestyle='dashed',alpha=0.5,color='lightgray')

#plt.plot(np.arange(100)*-1 +100,all_rt[0,:],color='b',alpha=0.3,linewidth=2)
plt.plot(x_data,all_rt[0,:],color='red',alpha=1,linewidth=2)
plt.plot(x_data,all_rt[1,:],color='red',alpha=1,linewidth=2)

plt.plot(x_data,nat_rt[0,:],color='b',alpha=1,linewidth=2)
plt.plot(x_data,nat_rt[1,:],color='b',alpha=1,linewidth=2)
#plt.plot(x_data,y_data)
plt.hlines(y=obs_data,xmin=0,xmax=100,linestyle='dashed')

# Calculate FAR

nat_rt_l=nat_rt[0,:]
nat_rt_u=nat_rt[1,:]
all_rt_l=all_rt[0,:]
all_rt_u=all_rt[1,:]

nat_l_prob = np.where(obs_data < nat_rt_l)
nat_u_prob = np.where(obs_data < nat_rt_u)
all_l_prob = np.where(obs_data < all_rt_l)
all_u_prob = np.where(obs_data < all_rt_u)

FAR1_l = 1 - (len(nat_rt_l[nat_l_prob])/len(all_rt_l[all_l_prob]))
FAR1_u = 1 - (len(nat_rt_u[nat_u_prob])/len(all_rt_u[all_u_prob]))

print("Mortality FAR method 1 ranges are:",FAR1_l,FAR1_u)

FAR2_l = 1 - (len(nat_rt_l[nat_l_prob])/len(all_rt_u[all_u_prob]))
FAR2_u = 1 - (len(nat_rt_u[nat_u_prob])/len(all_rt_l[all_l_prob]))

print("Mortality FAR method 1 ranges are:",FAR2_l,FAR2_u)







nat_rt_l_t=nat_rt_t[0,:]
nat_rt_u_t=nat_rt_t[1,:]
all_rt_l_t=all_rt_t[0,:]
all_rt_u_t=all_rt_t[1,:]

nat_l_prob_t = np.where(obs_data_t < nat_rt_l_t)
nat_u_prob_t = np.where(obs_data_t < nat_rt_u_t)
all_l_prob_t = np.where(obs_data_t < all_rt_l_t)
all_u_prob_t = np.where(obs_data_t < all_rt_u_t)

FAR1_l_t = 1 - (len(nat_rt_l_t[nat_l_prob_t])/len(all_rt_l_t[all_l_prob_t]))
FAR1_u_t = 1 - (len(nat_rt_u_t[nat_u_prob_t])/len(all_rt_u_t[all_u_prob_t]))

print("Temp FAR method 1 ranges are:",FAR1_l_t,FAR1_u_t)

FAR2_l_t = 1 - (len(nat_rt_l_t[nat_l_prob_t])/len(all_rt_u_t[all_u_prob_t]))
FAR2_u_t = 1 - (len(nat_rt_u_t[nat_u_prob_t])/len(all_rt_l_t[all_l_prob_t]))

print("Temp FAR method 1 ranges are:",FAR2_l_t,FAR2_u_t)




plt.xscale('log')
plt.title('London heat-related mortality (2006)',fontsize=15)
plt.xlabel('Return Period (yrs)',fontsize=10)
plt.ylabel('Max Daily Mortality',fontsize=10)
plt.xlim([0,100])
plt.ylim([0,100])

plt.savefig('london2006_mortality_v2.pdf')
plt.show()
