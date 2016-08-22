from __future__ import division
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
from pandas import HDFStore

np.set_printoptions(precision=5)
pd.set_option('display.expand_frame_repr', True)
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

def KalmanFilter(a1, p1, d, T, Z, Q, H, y):	
	m = y.shape[0]
	N = y.shape[1]
	f = a1.shape[0]

	a = np.zeros([f,N+1])
	a[:,0] = a1.flatten()
	P = np.zeros([N+1,f,f])
	P[0] = p1

	v = np.zeros([m, N])
	F = np.zeros([N, m, m])
	L = np.zeros([N, f, f])
	K = np.zeros([N, f, m])

	for i in range(0,N):
		v[:,i] = y[:,i]- np.dot(Z, a[:,i]).ravel()
		F[i] = np.dot(np.dot(Z,P[i]),Z.T)+H

		invF = np.linalg.inv(F[i])
		K[i] = np.dot(np.dot(np.dot(T,P[i]),Z.T),invF)

		L[i] = T - np.dot(K[i],Z)

		a[:,i+1] = np.dot(T,a[:,i]) + np.dot(K[i],v[:,i]) + d.reshape(3)
		P[i+1] = np.dot(np.dot(T,P[i]),T.T) + Q-np.dot(np.dot(K[i],F[i]),K[i].T)
	return a[:,1:N+1], P[1:N+1,:,:], v, F, K, L



def KalmanSmooth(a1,p1,d, T,Z, Q, H, y):
	a,P,v,F,K, L = KalmanFilter(a1, p1, d, T, Z, Q, H, y)

	f     = a1.shape[0]
	m     = v.shape[0]
	N     = v.shape[1]
	r     = np.zeros([f, 1])

	Alpha = np.zeros([f, N])
	for i in range(N-1,-1,-1):
		r = np.dot(np.dot(Z.T,inv(F[i])),v[:,i]).reshape(f,1)+ np.dot(L[i].T,r)
		#r = np.dot(Z.T, np.linalg.solve(F[i],v[:,i])).reshape(3,1) + np.dot(L[i].T,r)
		Alpha[:,i] = a[:,i] + np.dot(P[i],r).ravel() + d.ravel()
	return Alpha


def KFMVlik(a1,p1,d,T, Z, Q, H, y):
	m = y.shape[0]
	N = y.shape[1]

	a = a1
	P = p1

	logdetFsum = 0
	invFVVsum  = 0
	for i in range(0,N):
		v = y[:,i].reshape(m,1)- np.dot(Z, a)
		
		F = np.dot(np.dot(Z,P),Z.T)+H

		logdetF = np.log(np.linalg.det(F))
		logdetFsum += logdetF

		invFVV = np.dot(v.T,np.linalg.solve(F,v))
		invFVVsum += invFVV

		TPZ = np.dot(np.dot(T,P),Z.T)
		# K = TPZ*inv(F)
		KT = np.linalg.solve(F,TPZ.T)
		K = KT.T
		# a = a-d
		a = np.dot(T,a) + np.dot(K,v) + d
		
		P = np.dot(np.dot(T,P),T.T) + Q-np.dot(np.dot(K,F),K.T)

	return -0.5*N*m*np.log(2*np.pi) - 0.5*logdetFsum - 0.5*invFVVsum


def DFMlik(a1,p1,d,T, Z, Q, H, y):
	m = y.shape[0]
	N = y.shape[1]
	r = a1.shape[0]

	a1 = np.asmatrix(a1)
	p1 = np.asmatrix(p1)
	T = np.asmatrix(T)
	Z = np.asmatrix(Z)
	Q = np.asmatrix(Q)
	H = np.asmatrix(H)
	y = np.asmatrix(y)

	ahat = (Z.T*H.I*Z).I*Z.T*H.I*y

	C = (Z.T*H.I*Z).I
	newZ = np.asmatrix(np.eye(r))
	ahatlikval = KFMVlik(a1,p1,d,T,newZ,Q,C,ahat)
	
	detC = det(C)
	detH = det(H)

	E = y-Z*ahat
	vFv = 0
	invH = inv(H)

	vFv = np.sum(np.multiply(np.dot(invH.T,E), E))

	NegLoglik = 0.5*N*(m-r)*np.log(2*np.pi) -ahatlikval + 0.5*vFv+0.5*N*np.log(detH/detC)
	return NegLoglik

def TransformParBack24(pars):
	H = np.diag(np.exp(pars[0:24]))
	T = pars[24:33].reshape(3,3)

	mD = np.diag(np.exp(pars[33:36]))
	mL = np.eye(3)
	mL[1,0] = pars[36]
	mL[2,0] = pars[37]
	mL[2,1] = pars[38]
	Q = np.dot(np.dot(mL,mD),mL.T)

	dtemp = pars[39:42].reshape(3,1)
	d = np.dot((np.eye(3) - T), dtemp)

	Z1 = pars[42:66].reshape(8,3)
	Z2 = np.array([1,-1,1])
	Z3 = pars[66:72].reshape(2,3)
	Z4 = np.array([[1,1,1],[1,-1,-1]])
	Z5 = pars[72:78].reshape(2,3)
	Z6 = np.array([1,1,-1])
	Z7 = pars[78:102].reshape(8,3)
	Z = np.vstack([Z1,Z2,Z3,Z4,Z5,Z6,Z7])

	return H, T, Q, d, Z

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

#####################################################################################
####### MAIN ####### MAIN ####### MAIN ####### MAIN ####### MAIN ####### MAIN #######
#####################################################################################



# TICKERS = ['AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DJX', 'GE', 'GS',
#        'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
#        'MSFT', 'NKE', 'PFE', 'PG', 'SPX', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM'] 
# TICKERS = ['SPX']
# GROUP = 18
# SNAP = range(0,3430,10)
# for ticker in TICKERS:
# 	print 'Plotting', ticker
# 	VOLSOURCE = '/home/michael/Documents/Options/DATA/Vol/Vol_'+str(ticker)+'_'+str(GROUP)+'.csv'
# 	Vol = pd.read_csv(VOLSOURCE,header=None)
# 	mY = Vol.values.T
# 	N = mY.shape[0]
# 	T = mY.shape[1]
# 	F = 3

# 	PARSOURCE = '/home/michael/Documents/Options/DATA/ParInit/Par_'+str(ticker)+'_'+str(N)+'.csv'
# 	par  = pd.read_csv(PARSOURCE,header=None)

# 	pars = par.values.reshape(78)
# 	mH, mT, mQ, vd, mZ = TransformParBack18(pars)

# 	a1 = np.array([[0.20],[0.0021,],[-0.001]])
# 	p1 = np.eye(F)*0.1
# 	print '\nThe standard likelihood evaluated at current pars is:',float(KFMVlik(a1, p1, vd, mT, mZ, mQ, mH, mY))
# 	print 'The JKoopman likelihood evaluated at current pars is:'  ,float(-DFMlik(a1, p1, vd, mT, mZ, mQ, mH, mY))

# 	a,P,v,F,K,L= KalmanFilter(a1, p1, vd, mT, mZ, mQ, mH, mY)
# 	# Alpha = KalmanSmooth(a1, p1, vd, mT, mZ, mQ, mH, mY)


# # 	if ticker == 'SPX':
# # 		plt.plot(a[0,SNAP],label='Level of curve' + ticker, color = 'red', linewidth=0.7)
# # 		plt.plot(a[1,SNAP],label='Term Structure' + ticker, color = 'blue', linewidth=0.7)
# # 		plt.plot(a[2,SNAP],label='Smile Factor' + ticker, color = '#ff00ff', linewidth=0.7)
# # 	else:
# # 		plt.plot(a[0,SNAP], color = '#c0c0c0', linewidth=0.2)
# # 		plt.plot(a[1,SNAP], color = '#c0c0c0', linewidth=0.2)
# # 		plt.plot(a[2,SNAP], color = '#c0c0c0', linewidth=0.2)
# # 		plt.legend(loc='upper left',fontsize=10)
# # plt.show()



# Fit = np.dot(mZ, a)
# Residual = mY - Fit

# FitPlot = 0
# if FitPlot:
# 	for i in range(N):
# 		snap = range(0,T,5)
# 		plt.plot(mY[i,snap]       ,color='red'  ,label='Actual Vol',linewidth=0.3)
# 		plt.plot(Fit[i,snap]      ,color='blue' ,label='Fitted Vol',linewidth=0.3)
# 		plt.plot(Residual[i,snap] ,color='green',label='Residual'  ,linewidth=0.3)
# 		plt.legend(loc='upper left',fontsize=7)
# 		plt.title('General Dynamic Factors')
# 		plt.show()