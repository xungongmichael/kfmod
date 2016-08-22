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
import ctypes as ct 
from numpy.linalg import det, inv
import os 
import timeit
import numpy.ma as ma
import matplotlib.pylab as plt
#os.chdir('/home/michael/Desktop/PyKalman')
sys.path.append('/home/michael/Documents/Options/SRC/PyKalman')
import utils._checkers as _checkers
import utils.Utils as utils
import utils.Kalman as kl
import core.cython.kfpack as kf
import importlib
importlib.reload(_checkers)
importlib.reload(utils)


class _plot:
	def __init__(self,filter_mean,filter_var,predict_mean,predict_var,
			smooth_mean,smooth_var):
		self._filter_mean = filter_mean 
		self._filter_var = filter_var 
		self._predict_mean = predict_mean
		self._predict_var = predict_var
		self._smooth_mean = smooth_mean
		self._smooth_var = smooth_var
		self.n_dim_state = filter_mean.shape[0]

	def filter(self,stat='mean', title=None, state = None, label=None, 
			show=True, savefig=False, save_path=None,figsize=None,linewidth=0.3):
		try:
			import matplotlib.pylab as plt
			plt.style.use('ggplot')
		except ImportError as e:
			raise e
		if label is None:
			_label = ['State '+str(i+1) for i in range(self.n_dim_state)]
		else:
			_label = label

		if title is None:
			_title = 'kalman filtered '+ stat 
		else:
			_title = title 

		if state is None:
			_state = range(self.n_dim_state)
		else:
			_state = state

		fig = plt.figure(num=None, figsize=figsize, dpi=300, 
											facecolor='w', edgecolor='k')
		if stat is 'mean':
			ax = fig.add_subplot(111)
			for i in _state:
				ax.plot(self._filter_mean.T[:,i],label=_label[i],linewidth=linewidth)
		if stat is 'variance':
			ax = fig.add_subplot(111)
			variance = np.zeros_like(self._filter_mean)
			for i in range(variance.shape[1]):
				variance[:,i] = np.diagonal(self._filter_var[i])
			for i in _state:
				ax.plot(variance.T[:,i],label=_label[i],linewidth=linewidth)	

		plt.title(_title)
		plt.legend()

		if savefig:
			if save_path is not None:
				plt.savefig(save_path +'/filter_mean.eps')
			else:
				plt.savefig('filter_mean.eps')
		if save_path is not None:
			plt.savefig(save_path +'/filter_mean.eps')
		if show:
			plt.show()
		return fig

	def predict(self,stat='mean', title=None, state = None, label=None, 
			show=True, savefig=False, save_path=None,figsize=None,linewidth=0.3):
		try:
			import matplotlib.pylab as plt
			plt.style.use('ggplot')
		except ImportError as e:
			raise e
		if label is None:
			_label = ['State '+str(i+1) for i in range(self.n_dim_state)]
		else:
			_label = label

		if title is None:
			_title = 'kalman predicted '+ stat 
		else:
			_title = title 

		if state is None:
			_state = range(self.n_dim_state)
		else:
			_state = state

		fig = plt.figure(num=None, figsize=figsize, dpi=300, 
											facecolor='w', edgecolor='k')
		if stat is 'mean':
			ax = fig.add_subplot(111)
			for i in _state:
				ax.plot(self._predict_mean.T[:,i],label=_label[i],linewidth=linewidth)
		if stat is 'variance':
			ax = fig.add_subplot(111)
			variance = np.zeros_like(self._predict_mean)
			for i in range(variance.shape[1]):
				variance[:,i] = np.diagonal(self._predict_var[i])
			for i in _state:
				ax.plot(variance.T[:,i],label=_label[i],linewidth=linewidth)		

		plt.title(_title)
		plt.legend()

		if savefig:
			if save_path is not None:
				plt.savefig(save_path +'/filter_mean.eps')
			else:
				plt.savefig('filter_mean.eps')
		if save_path is not None:
			plt.savefig(save_path +'/filter_mean.eps')
		if show:
			plt.show()
		return fig

	def smooth(self,stat='mean', title=None, state = None, label=None, 
			show=True, savefig=False, save_path=None,figsize=None,linewidth=0.3):
		try:
			import matplotlib.pylab as plt
			plt.style.use('ggplot')
		except ImportError as e:
			raise e
		if label is None:
			_label = ['State '+str(i+1) for i in range(self.n_dim_state)]
		else:
			_label = label

		if title is None:
			_title = 'kalman smoothed '+ stat 
		else:
			_title = title 

		if state is None:
			_state = range(self.n_dim_state)
		else:
			_state = state

		fig = plt.figure(num=None, figsize=figsize, dpi=300, 
											facecolor='w', edgecolor='k')
		if stat is 'mean':
			ax = fig.add_subplot(111)
			for i in _state:
				ax.plot(self._smooth_mean.T[:,i],label=_label[i],linewidth=linewidth)
		if stat is 'variance':
			ax = fig.add_subplot(111)
			variance = np.zeros_like(self._smooth_var)
			for i in range(variance.shape[1]):
				variance[:,i] = np.diagonal(self._smooth_var[i])
			for i in _state:
				ax.plot(variance.T[:,i],label=_label[i],linewidth=linewidth)		

		plt.title(_title)
		plt.legend()

		if savefig:
			if save_path is not None:
				plt.savefig(save_path +'/filter_mean.eps')
			else:
				plt.savefig('filter_mean.eps')
		if save_path is not None:
			plt.savefig(save_path +'/filter_mean.eps')
		if show:
			plt.show()
		return fig
