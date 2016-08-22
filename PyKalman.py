"""
=====================================
Inference for Linear-Gaussian Systems
=====================================

This module implements the Kalman Filter, Kalman Smoother, and
Likelihood Maximization Algorithm for Linear-Gaussian state space models
"""
import numpy as np 
import pandas as pd 
import warnings
import sys
from numpy.linalg import det, inv
import os 
import timeit
import time
import numpy.ma as ma
import matplotlib.pylab as plt
sys.path.append('/home/michael/Documents/Options/SRC/PyKalman')
import utils._checkers as _checkers
import utils.Utils as utils
import core.cython.kfpack as kf
import Visualization
import importlib
importlib.reload(Visualization)
from Visualization import _plot
importlib.reload(_checkers)
importlib.reload(utils)


class KalmanFilter:
	def __init__(self, mY , vC = None, mZ = None, mH = None, 
		vD = None, mT = None, mQ = None, a1 = None, p1 = None,
		n_r = None):

		_checkers._input_dim_checker(mY,vC,mZ,mH,vD,mT,mQ,a1,p1,n_r)

		mY,vC,mZ,mH,vD,mT,mQ,a1,p1 = _checkers._fortran_checker(mY,
			vC,mZ,mH,vD,mT,mQ,a1,p1)

		if np.any(np.isnan(mY)):
			raise NotImplementedError ('''Observations mY contains 
				missing values is not implemented''')
		self._mY = mY
		self._vC = vC
		self._mZ = mZ 
		self._mH = mH 
		self._vD = vD 
		self._mT = mT 
		self._mQ = mQ
		self.n_r = n_r
		self.n_m = mY.shape[0]
		self.n_T = mY.shape[1]

		#Decompose Q by LDL for estimation
		L,D = utils.LDL(self._mQ)
		self._mQ_L = L
		self._mQ_D = D

		"""Initialize Constraint Data"""
		self._vC_ma = np.zeros_like(self._vC)
		self._mZ_ma = np.zeros_like(self._mZ)
		self._mH_ma = np.zeros_like(self._mH)
		self._vD_ma = np.zeros_like(self._vD)
		self._mT_ma = np.zeros_like(self._mT)
		self._mQ_ma = np.zeros_like(self._mQ)

		self._mQ_L_ma = np.zeros_like(self._mQ_L)
		self._mQ_L_ma[np.triu_indices(self.n_r)] = 1

		self._mQ_D_ma = np.zeros_like(self._mQ_D)
		self._mQ_D_ma[np.triu_indices(self.n_r,1)] = 1
		self._mQ_D_ma[np.tril_indices(self.n_r,-1)] = 1

		"""Runtime info"""
		self._estimated = 0
		self._diag_mH = False
		self._diag_mQ = False

		if a1 is None or a1 is "Diffuse":
			self.a1 = np.asfortranarray(
					np.dot(inv(np.eye(n_r)-mT),vD))
		else:
			self.a1 = a1

		if p1 is None:
			self.p1 = np.asfortranarray(np.dot(
				inv(np.eye(n_r**2)-np.kron(mT,mT)),
				mQ.reshape(n_r**2,1)).reshape(n_r,n_r))
		elif p1 == 'Diffuse':
			self.p1 = np.asfortranarray(np.eye(n_r)*1E8)
		else:
			self.p1 = p1

	def set_constraint(self,
		vC_cons = None, mZ_cons = None, mH_cons = None, 
		vD_cons = None, mT_cons = None, mQ_cons = None, 
		mQ_L_cons = None, mQ_D_cons = None, 
		diag_mH= False, diag_mQ=False):

		try:
			_checkers._input_dim_checker(
				self._mY,vC=vC_cons,mZ=mZ_cons,
				mH=mH_cons,vD=vD_cons,mT=mT_cons,
				mQ=mQ_cons,n_r = self.n_r)
		except TypeError:
			print('''Dimentions of constraint data does not 
				match the input parameters''')
		
		self._vC_cons = False
		self._mZ_cons = False
		self._vD_cons = False
		self._mT_cons = False

		if vC_cons is not None:
			self._vC_ma = vC_cons 
		if mZ_cons is not None:
			self._mZ_ma = mZ_cons
		if mH_cons is not None:
			self._mH_ma = mH_cons
		else:
			if diag_mH:
				self._diag_mH = True
				self._mH_ma = np.ones([self.n_m,self.n_m])
				np.fill_diagonal(self._mH_ma,0)
		if vD_cons is not None:
			self._vD_ma = vD_cons
		if mT_cons is not None:
			self._mT_ma = mT_cons
		if mQ_cons is not None: 
			self._mQ_ma = mQ_cons
		else:
			if diag_mQ:
				self._diag_mQ = True
				self._mQ_L = np.eye(self.n_r)
				self._mQ_L_ma = np.ones_like(self._mQ_L)
				np.fill_diagonal(self._mQ_L_ma,0)
		if mQ_L_cons is not None:
			self._mQ_L_ma = mQ_L_cons
		if mQ_D_cons is not None:
			self._mQ_D_ma = mQ_L_cons

	def _disassembler(self):
		"""Disassember the parameter matrices to vector"""
		vC_masked 	= ma.array(self._vC,   mask=self._vC_ma)
		mZ_masked 	= ma.array(self._mZ,   mask=self._mZ_ma)
		mH_masked 	= ma.array(self._mH,   mask=self._mH_ma)
		vD_masked 	= ma.array(self._vD,   mask=self._vD_ma)
		mT_masked 	= ma.array(self._mT,   mask=self._mT_ma)
		mQ_L_masked = ma.array(self._mQ_L, mask=self._mQ_L_ma)
		mQ_D_masked = ma.array(self._mQ_D, mask=self._mQ_D_ma)


		vC_par 	 = vC_masked.flatten('F').compressed()
		mZ_par 	 = mZ_masked.flatten('F').compressed()
		mH_par 	 = np.log(mH_masked.flatten('F').compressed())
		vD_par 	 = vD_masked.flatten('F').compressed()
		mT_par 	 = mT_masked.flatten('F').compressed()
		mQ_par_L = mQ_L_masked.flatten('F').compressed()
		mQ_par_D = np.log(mQ_D_masked.flatten('F').compressed())

		
		self._starting_value = np.hstack([vC_par,mZ_par,mH_par,\
			vD_par,mT_par,mQ_par_L,mQ_par_D])

		"""Disassember information"""
		self._maskinfo = np.hstack([
			self._vC_ma.flatten('F'),
			self._mZ_ma.flatten('F'),
			self._mH_ma.flatten('F'),self._vD_ma.flatten('F'),
			self._mT_ma.flatten('F'),self._mQ_L_ma.flatten('F'),
			self._mQ_D_ma.flatten('F')]).astype(np.int32)

		self._parminfo = np.hstack([
			self._vC.flatten('F'),
			self._mZ.flatten('F'),self._mH.flatten('F'),
			self._vD.flatten('F'),self._mT.flatten('F'),
			self._mQ_L.flatten('F'),self._mQ_D.flatten('F')])

	def _assembler(self):
		j = 0
		param = np.zeros_like(self._parminfo)
		for i,mask in enumerate(self._maskinfo):
			if mask:
				param[i] = self._parminfo[i]
			else:
				param[i] = self._optimized_param[j]
				j += 1
		m = self.n_m
		r = self.n_r 

		ivC = m
		imZ = ivC + m*r
		imH = imZ + m**2
		ivD = imH + r
		imT = ivD + r**2
		imQL= imT + r**2
		imQD= imQL+ r**2

		self._vC = param[ 0 :ivC].reshape([m,1],order='F')
		self._mZ = param[ivC:imZ].reshape([m,r],order='F')
		self._mH = param[imZ:imH].reshape([m,m],order='F')
		Hidx = np.diag_indices(m)
		self._mH[Hidx] = np.exp(self._mH[Hidx])

		self._vD = param[imH:ivD].reshape([r,1],order='F')
		self._mT = param[ivD:imT].reshape([r,r],order='F')
		self._mQ_L = param[imT:imQL].reshape([r,r],order='F')

		self._mQ_D = param[imQL:imQD].reshape([r,r],order='F')
		Qidx = np.diag_indices(r)
		self._mQ_D[Qidx] = np.exp(self._mQ_D[Qidx])
		self._mQ = np.asfortranarray(
				self._mQ_L.dot(self._mQ_D).dot(self._mQ_L.T))

	def _analytical_gradient_experimental(self):
		# self._disassembler()
		# par_to_est = self._starting_value 

		# self._score = sc.gradient(par_to_est,self._maskinfo,
		# 	self._parminfo,self._mY,
		# 	self.n_r,self.n_m,self.n_T,
		# 	1,1,0,0,
		# 	self._vC_ma,self._mZ_ma,self._mH_ma,self._vD_ma,
		# 	self._mT_ma,self._mQ_L_ma,self._mQ_D_ma)

	def estimate(self,module='scipy',
		factr=1E3, MaxIter=10000,delta=0.001,
		epsilon=0.01,ftol=0.01,gtol=0.1,mhess=10,iPrint=1):
		
		self._disassembler()
		par_to_est = self._starting_value 

		if module == 'scipy':
			from scipy.optimize import fmin_l_bfgs_b
			def dfmopt(pars,*args):
				return kf.likelihood(pars,
					args[0],args[1],args[2],
					args[3],args[4],args[5])

			def dfmscr(pars,*args):
				return kf.cgradient(pars,
					args[ 0],args[ 1],args[ 2],
					args[ 3],args[ 4],args[ 5])
			
			r = self.n_r
			m = self.n_m
			N = self.n_T
			fmin = fmin_l_bfgs_b(dfmopt,par_to_est,fprime=dfmscr,pgtol=gtol,
				args=(self._maskinfo,self._parminfo,self._mY,r,m,N),
				factr=factr,iprint=iPrint,maxiter=MaxIter,maxfun=MaxIter)

			self._estimated = 1
			self._optimized_param = fmin[0]
			self._likelihood = fmin[1]
			self._assembler()

		elif module == 'C++':
			self._likelihood = 0
			retcode = kf.estimate(par_to_est,self._maskinfo,self._parminfo,
						self._mY,self.n_r,self._likelihood,
						MaxIter=MaxIter,delta=delta,epsilon=epsilon,
						ftol=ftol,gtol=gtol,mhess=mhess,iPrint=iPrint)

			print (utils.LBFGSretcode(retcode))
			self._estimated = 1
			self._optimized_param = par_to_est
			self._assembler()
		else:
			raise KeyError("module has to be scipy or C++")
		
	def _moment(self):
		self._filter_mean, _filter_var, self._predict_mean, _predict_var\
			= kf.moment(self.a1,self.p1,self._vC,self._mZ,self._mH,
									self._vD,self._mT,self._mQ,self._mY)

		self._filter_var = np.rollaxis(_filter_var.reshape(
					self.n_r,self.n_T,self.n_r),1,0)
		self._predict_var = np.rollaxis(_predict_var.reshape(
					self.n_r,self.n_T,self.n_r),1,0)

	def _smooth(self):
		self._filter_mean, _filter_var, self._predict_mean,\
						_predict_var, self._smooth_mean, _smooth_var\
			= kf.smooth(self.a1,self.p1,self._vC,self._mZ,self._mH,
								self._vD,self._mT,self._mQ,self._mY)

		self._filter_var = np.rollaxis(_filter_var.reshape(
					self.n_r,self.n_T, self.n_r),1,0)
		self._predict_var = np.rollaxis(_predict_var.reshape(
					self.n_r,self.n_T,self.n_r),1,0)
		self._smooth_var = np.rollaxis(_smooth_var.reshape(
					self.n_r,self.n_T,self.n_r),1,0)

	def _cal_likelihood(self):
		if hasattr(self,'_optimized_param'):
			vP = self._optimized_param
		else:
			self._disassembler()
			vP = self._starting_value

		self._likelihood = kf.likelihood(vP,self._maskinfo,
				self._parminfo,self._mY,self._maskinfo.shape[0],
				self.n_r, self.n_m,self.n_T)

	def print_parameter(self,string=None):
		if string is not None:
			if not isinstance(string, str):
				raise TypeError("request parameter should be put in string list")
			params = string.replace(',',' ').replace(';',' ').replace('.',' ').split(' ')
			has_ele = False
			for ele in params:
				if ele in ['vC','mZ','diag_mH','vD','mT','mQ','diag_mQ','a1','p1']:
					has_ele = True
					break
				else:
					raise KeyError('''request parameter should be in 
						['vC','mZ','diag_mH','vD','mT','mQ','diag_mQ',
						'a1','p1'], and separated by ; , . or blank''')
		else:
			params = ['vC','mZ','diag_mH','vD','mT','mQ','a1','p1']

		if 'vC' in params:
			print("Observation intercept (vC):\n", self.vC)
		if 'mZ' in params:
			print("Loading matirx (mZ):\n", self.mZ)
		if 'mH' in params or 'diag_mH' in params:
			if 'diag_mH' in params:
				print("Measurement error (mH) diagonal:\n", np.diagonal(self.mH))
			else:
				print("Measurement error (mH) matrix:\n", self.mH)
		if 'vD' in params:
			print("State intercept (vD):\n",self.vD)
		if 'mT' in params:
			print("State transistion matrix (mT):\n",self.mT)
		if 'mQ' in params or 'diag_mQ' in params:
			if 'diag_mQ' in params:
				print("State variance (mQ) diagonal:\n", np.diagonal(self.mQ))
			else:
				print("State variance (mQ):\n", self.mQ)
		if 'a1' in params:
			print("Initial states:\n", self.a1)
		if 'p1' in params:
			print("Initial states variance:\n",self.p1)

	#TODO: advanced overloading parameters
	def _overload_params(self,vP):
		if not vP.flags.f_contiguous:
			vP = np.asfortranarray(vP)
		self._optimized_param = vP
		self._disassembler()
		self._assembler()
		self.a1 = np.asfortranarray(
			np.dot(inv(np.eye(self.n_r)-self.mT),self.vD))
		self.p1 = np.asfortranarray(np.dot(
				inv(np.eye(self.n_r**2)-np.kron(self.mT,self.mT)),
				self.mQ.reshape(self.n_r**2,1)).reshape(self.n_r,self.n_r))
	#TODO: Diagnositc check
	def residual_check(self):
		try:
			from scipy.stats.mstats import normaltest
		except Exception as e:
			raise
		test = normaltest(self.residual.T)
		print (test[-1])

	def forecast_states(self,horizon = 1):
		self._moment()
		if horizon < 1:
			raise ValueError("forecast horizon can't be less than 1")
		elif horizon == 1:
			return self.predict_mean[:,-1].reshape(self.n_r,1)
		else:
			forecast_states = np.zeros([self.n_r, horizon])
			forecast_states[:,0] = self.predict_mean[:,-1]
			for i in range(1, horizon):
				forecast_states[:,i] = np.dot(self.mT,forecast_states[:,i-1])\
									 + self.vD.ravel()
			return forecast_states

	def forecast_obs(self,horizon = 1):
		self._moment()
		if horizon < 1:
			raise ValueError("forecast horizon can't be less than 1")
		elif horizon == 1:
			return np.dot(self.mZ,self.predict_mean[:,-1]).reshape(
												self.n_m,1) + self.vC
		else:
			forecast_obs = np.zeros([self.n_m, horizon])
			forecast_states = self.forecast_states(horizon)
			for i in range(0, horizon):
				forecast_obs[:,i] = np.dot(self.mZ,forecast_states[:,i]) \
								  + self.vC.ravel()
			return forecast_obs

	def Update_mY(self,mY):
		if not isinstance(mY, np.ndarray):
			raise TypeError("Input observations mY is not numpy object!")
		if mY.shape[0] > mY.shape[1]:
			warnings.warn('''Input observations mY seems have more 
				dimensions than sample length, make sure mY is a 
				column matrix''', RuntimeWarning)
		if mY.shape[0] != self.n_m:
			raise TypeError('''Update mY's dimention does not match 
				bservations dimention''')
		if not mY.flags.f_contiguous:
			mY = np.asfortranarray(mY)
		self.n_m = mY.shape[0]
		self.n_T = mY.shape[1]	
		self._mY = mY

	@property
	def likelihood(self):
		if hasattr(self,'_likelihood'):
			return -self._likelihood
		else:
			self._cal_likelihood()
			return -self._likelihood

	@property
	def filter(self):
		if hasattr(self,'_filter_mean'):
			return [self._filter_mean, self._filter_var]
		else:
			self._moment()
			return [self._filter_mean, self._filter_var]

	@property
	def predictor(self):
		if hasattr(self,'_predict_mean'):
			return [self._predict_mean, self._predict_var]
		else:
			self._moment()
			return [self._predict_mean, self._predict_var]

	@property
	def filter_mean(self):
		if hasattr(self,'_predict_mean'):
			return self._filter_mean
		else:
			self._moment()
			return self._filter_mean
	
	@property
	def filter_var(self):
		if hasattr(self,'_filter_var'):
			return self._filter_var
		else:
			self._moment()
			return self._filter_var

	@property
	def predict_mean(self):
		if hasattr(self,'_predict_mean'):
			return self._predict_mean
		else:
			self._moment()
			return self._predict_mean	
	
	@property
	def predict_var(self):
		if hasattr(self,'_predict_var'):
			return self._predict_var
		else:
			self._moment()
			return self._predict_var

	@property
	def smooth_mean(self):
		if hasattr(self,'_smooth_mean'):
			return self._smooth_mean
		else:
			self._smooth()
			return self._smooth_mean	
	
	@property
	def smooth_var(self):
		if hasattr(self,'_smooth_var'):
			return self._smooth_var
		else:
			self._smooth()
			return self._smooth_var	

	@property 
	def plot(self):
		visual = _plot(self.filter_mean,self.filter_var,
			self.predict_mean,self.predict_var,
			self.smooth_mean,self.smooth_var)
		return visual

	@property 
	def residual(self):
		return self.mY-np.dot(self.mZ,self.smooth_mean)


	@property
	def mY(self):
		return self._mY

	@property
	def vC(self):
		return self._vC

	@property
	def mZ(self):
		return self._mZ

	@property
	def mH(self):
		return self._mH

	@property
	def vD(self):
		return self._vD

	@property
	def mT(self):
		return self._mT

	@property
	def mQ(self):
		return self._mQ

