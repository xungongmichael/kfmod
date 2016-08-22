import numpy as np 
import pandas as pd 
import warnings
import sys
from numpy.linalg import det, inv
import os 
import timeit
import numpy.ma as ma
import time
sys.path.append('/home/michael/Documents/Options/SRC/PyKalman')
sys.path.append("/home/michael/Documents/Options/SRC/PyKalman/core/cython")
import utils._checkers as _checkers
import core.PyAPI as PyAPI
import utils.Utils as utils
import utils.Kalman as kl
import matplotlib.pylab as plt
import kfpack as kf 
import PyKalman as pk 
from scipy.optimize import minimize,fmin_l_bfgs_b
reload(_checkers)
reload(utils)
reload(PyAPI)


def TransformParBack18(pars):
	H = np.diag(np.exp(pars[0:18]))
	T = pars[18:27].reshape(3,3)

	mD = np.diag(np.exp(pars[27:30]))
	mL = np.eye(3)
	mL[1,0] = pars[30]
	mL[2,0] = pars[31]
	mL[2,1] = pars[32]
	Q = np.dot(np.dot(mL,mD),mL.T)

	dtemp = pars[33:36].reshape(3,1)
	d = np.dot((np.eye(3) - T), dtemp)

	Z1 = pars[36:54].reshape(6,3)
	Z2 = np.array([1,-1,1])
	Z3 = pars[54:57].reshape(1,3)
	Z4 = np.array([[1,1,1],[1,-1,-1]])
	Z5 = pars[57:60].reshape(1,3)
	Z6 = np.array([1,1,-1])
	Z7 = pars[60:78].reshape(6,3)
	Z = np.vstack([Z1,Z2,Z3,Z4,Z5,Z6,Z7])

	return H, T, Q, d, Z

mY = np.ascontiguousarray(pd.read_csv('/home/michael/Documents/Options/SRC/PyKalman/test/Vol_SPX_18.csv').values.T)

m = 18
r = 3
N = mY.shape[1]
par  = pd.read_csv('/home/michael/Documents/Options/SRC/PyKalman/test/Par_SPX_18_opt.csv',header=None)
pars = par.values.reshape(78)
mH, mT, mQ, vD, mZ = TransformParBack18(pars)
vC = np.zeros([m,1])
mZ_cons = np.zeros_like(mZ)
mZ_cons[[6,8,9,11],:] = 1
vC_cons = np.ones([m,1])





# KF = pk.KalmanFilter(mY, vC=vC, mZ=mZ, mH=mH, vD=vD, mT=mT, mQ=mQ, n_dim_state=r)
# KF.set_constraint(vC_cons=vC_cons,mZ_cons = mZ_cons,diag_mH=True)
# KF._disassembler()

# val = kf.likelihood(KF._starting_value,KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N)
# a = kf.gradient(KF._starting_value,KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N)
# b = kf.cgradient(KF._starting_value,KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N)

# '''test moment'''
# a1 = np.zeros([r,1])
# p1 = np.eye(r)
# ae1,b1,c1,d1 = kf.moment(a1,p1,vC,mZ,mH,vD,mT,mQ,mY)
# a,b,c,d,e,f = kf.smooth(a1,p1,vC,mZ,mH,vD,mT,mQ,mY)

# def dfmopt(pars,*args):
# 	vMask = args[0]
# 	vPams = args[1]
# 	mY = args[2]
# 	m_mask = args[3]
# 	r = args[4]
# 	m = args[5]
# 	N = args[6]

# 	return kf.likelihood(pars,vMask,vPams,mY,m_mask,r,m,N)

# def dfmscr(pars,*args):
# 	vMask = args[0]
# 	vPams = args[1]
# 	mY = args[2]
# 	m_mask = args[3]
# 	r = args[4]
# 	m = args[5]
# 	N = args[6]

# 	return kf.cgradient(pars,vMask,vPams,mY,m_mask,r,m,N)
# t0=time.time()
# for i in range(1000):
# 	b = kf.cgradient(KF._starting_value,KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N)
# print 'Finished in:',(time.time()-t0)/1000, 'Seconds'
# # dfmR = minimize(dfmopt, KF._starting_value, args=(KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N),jac=dfmscr,method='L-BFGS-B',options={'disp': True,'maxiter': 20000,'factr':1e2})

# dfmR = fmin_l_bfgs_b(dfmopt,KF._starting_value,fprime=dfmscr,args=(KF._maskinfo,KF._parminfo,KF._mY,KF._maskinfo.shape[0],r,m,N),factr=1e3,iprint=1,maxiter=20)
# x = dfmR[0]

# for i in range(1000, 2000):
# 	Y = mY[:,i-1000:i].copy()
# 	dfmR = fmin_l_bfgs_b(dfmopt,x,fprime=dfmscr,args=(KF._maskinfo,KF._parminfo,Y,KF._maskinfo.shape[0],r,m,N),factr=1e3,iprint=1,maxiter=20)
# 	x = dfmR[0]
# 	print x.flags
# 	time.sleep(1)

	#time.sleep(1)

# nPar = KF._starting_value.shape[0]
# m_mask = KF._maskinfo.shape[0]
# fx = 0
# kf.estimate(KF._starting_value, KF._maskinfo, KF._parminfo, KF._mY,nPar,m_mask,r,m,N,fx,iPrint=1,gtol=1e-05)
# print "fx:", fx
# print "norm:",np.linalg.norm(KF._starting_value)
