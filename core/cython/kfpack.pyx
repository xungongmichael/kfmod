import numpy as np 
cimport numpy as np 
import cython

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

from libc.math cimport *

cdef extern from "kfpack.h":
	cdef cppclass kf:
		void Moment(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, double *p_mH, double *p_vD, double *p_mT, double *p_mQ, double *p_mY, double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, double *p_mP_predictor, int r, int m, int N) nogil
		void Smooth(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, double *p_mH, double *p_vD,double *p_mT, double *p_mQ, double *p_mY, double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, double *p_mP_predictor, double *p_mA_smoother, double *p_mP_smoother,int r, int m, int N) nogil

		double MICevallik(double *vPar, int *vMask, double *vPams, double *p_mY, int m_mask, int r, int m, int N) nogil

		int NumGradient(double *vScore, double *vPar, int *vMask, double *vPams, double *p_mY, int m_mask, int numPars, int r, int m, int N) nogil

		int callLBFGS(double *vPar, int *vMask, double *vPams, double *data, int n_par, int m_mask, int r, int m, int N, double *fx, int MaxIter, double delta, double epsilon, double ftol, double gtol, int mhess, int iPrint) nogil
 
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def moment( np.ndarray[double, ndim=2, mode="fortran"] a1,
			np.ndarray[double, ndim=2, mode="fortran"] p1,
			np.ndarray[double, ndim=2, mode="fortran"] vC,
			np.ndarray[double, ndim=2, mode="fortran"] mZ,
			np.ndarray[double, ndim=2, mode="fortran"] mH,
			np.ndarray[double, ndim=2, mode="fortran"] vD,
			np.ndarray[double, ndim=2, mode="fortran"] mT,
			np.ndarray[double, ndim=2, mode="fortran"] mQ,
			np.ndarray[double, ndim=2, mode="fortran"] mY):
	cdef:
		int r = a1.shape[0]
		int m = mY.shape[0]
		int N = mY.shape[1]
		np.ndarray[double, ndim=2, mode="fortran"] \
								mAF = np.zeros([r,N],order='F')
		np.ndarray[double, ndim=2, mode="fortran"] \
								mPF = np.zeros([r,N*r],order='F')
		np.ndarray[double, ndim=2, mode="fortran"] \
								mAP = np.zeros([r,N],order='F')
		np.ndarray[double, ndim=2, mode="fortran"] \
								mPP = np.zeros([r,N*r],order='F')	
		kf kfpack 
	kfpack.Moment(
		&a1[0,0],&p1[0,0],&vC[0,0],&mZ[0,0],
		&mH[0,0],&vD[0,0],&mT[0,0],&mQ[0,0],&mY[0,0],
		&mAF[0,0],&mPF[0,0],&mAP[0,0],&mPP[0,0],r,m,N)

	return mAF, mPF, mAP, mPP


def smooth( np.ndarray[double, ndim=2, mode="fortran"] a1,
			np.ndarray[double, ndim=2, mode="fortran"] p1,
			np.ndarray[double, ndim=2, mode="fortran"] vC,
			np.ndarray[double, ndim=2, mode="fortran"] mZ,
			np.ndarray[double, ndim=2, mode="fortran"] mH,
			np.ndarray[double, ndim=2, mode="fortran"] vD,
			np.ndarray[double, ndim=2, mode="fortran"] mT,
			np.ndarray[double, ndim=2, mode="fortran"] mQ,
			np.ndarray[double, ndim=2, mode="fortran"] mY):
	cdef:
		int r = a1.shape[0]
		int m = mY.shape[0]
		int N = mY.shape[1]
		np.ndarray[double,ndim=2,mode="fortran"] \
						mAF = np.zeros([r,N],order='F')
		np.ndarray[double,ndim=2,mode="fortran"] \
						mPF = np.zeros([r,N*r],order='F')
		np.ndarray[double,ndim=2,mode="fortran"] \
						mAP = np.zeros([r,N+1],order='F')
		np.ndarray[double,ndim=2,mode="fortran"] \
						mPP = np.zeros([r,(N+1)*r],order='F')
		np.ndarray[double,ndim=2,mode="fortran"] \
						mAS = np.zeros([r,N],order='F')
		np.ndarray[double,ndim=2,mode="fortran"] \
						mPS = np.zeros([r,N*r],order='F')	
		kf kfpack 
	kfpack.Smooth(
		&a1[0,0],&p1[0,0],&vC[0,0],&mZ[0,0],
		&mH[0,0],&vD[0,0],&mT[0,0],&mQ[0,0],&mY[0,0],
		&mAF[0,0],&mPF[0,0],&mAP[0,0],&mPP[0,0],
		&mAS[0,0],&mPS[0,0],r,m,N)

	return mAF, mPF, mAP[:,1:], mPP[:,r:], mAS, mPS



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def likelihood( 
	np.ndarray[double, ndim=1, mode="fortran"] vPar,
	np.ndarray[int   , ndim=1, mode="fortran"] vMask,
	np.ndarray[double, ndim=1, mode="fortran"] vPams,
	np.ndarray[double, ndim=2, mode="fortran"] mY,
	int r, int m, int N):
	
	cdef:
		kf kfpack
		int m_mask = vMask.shape[0]
	return kfpack.MICevallik(
		&vPar[0],&vMask[0],&vPams[0],
		&mY[0,0],m_mask,r,m,N)


cdef double _EvalFunc(int i, double *avP, int *vMask, double *vPams, 
					double *mY, int m_mask, int r, int m, int N, int num) nogil:
	cdef:
		double p, fp, fm, jac, jac_eps 
		double * local_vP
		Py_ssize_t j
		kf kfpack

	local_vP = <double *> malloc(sizeof(double) * num)
	for j in xrange(num):
		local_vP[j] = avP[j]

	jac_eps = 5E-6
	p = local_vP[i]

	local_vP[i] = p + jac_eps
	fp = kfpack.MICevallik(local_vP, vMask, vPams, mY, m_mask, r, m, N)

	local_vP[i] = p - jac_eps 
	fm = kfpack.MICevallik(local_vP, vMask, vPams, mY, m_mask, r, m, N)

	local_vP[i] = p 
	jac = (fp-fm)/(2*jac_eps)

	free(local_vP)
	return jac

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def gradient(
	np.ndarray[double, ndim=1, mode="fortran"] vPar,
	np.ndarray[int   , ndim=1, mode="fortran"] vMask,
	np.ndarray[double, ndim=1, mode="fortran"] vPams,
	np.ndarray[double, ndim=2, mode="fortran"] mY,
	int r, int m, int N):
	cdef:
		double[:] score = np.zeros(vPar.shape[0])
		Py_ssize_t j
		int num = vPar.shape[0]
		int m_mask = vMask.shape[0]

	for j in prange(num, nogil=True):
		score[j] = _EvalFunc(j,&vPar[0],&vMask[0],\
			&vPams[0],&mY[0,0],m_mask,r,m,N,num)

	return np.asarray(score)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cgradient(
	np.ndarray[double, ndim=1, mode="fortran"] vPar,
	np.ndarray[int   , ndim=1, mode="fortran"] vMask,
	np.ndarray[double, ndim=1, mode="fortran"] vPams,
	np.ndarray[double, ndim=2, mode="fortran"] mY,
	int r, int m, int N):
	cdef:
		double[:] vScore = np.zeros(vPar.shape[0])
		Py_ssize_t j
		int num = vPar.shape[0]
		int m_mask = vMask.shape[0]
		kf kfpack

	kfpack.NumGradient(&vScore[0],&vPar[0],&vMask[0],\
		&vPams[0],&mY[0,0],m_mask,num,r,m,N)
	
	return np.asarray(vScore)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def estimate(
	np.ndarray[double, ndim=1, mode="fortran"] vPar,
	np.ndarray[int,    ndim=1, mode="fortran"] vMask,
	np.ndarray[double, ndim=1, mode="fortran"] vPams,
	np.ndarray[double, ndim=2, mode="fortran"] mY, 
	int r, double fx,int MaxIter=1000, double delta=0.0, 
	double epsilon=0.01,double ftol=0.01, double gtol=0.1,
	int mhess=10, int iPrint=1):
	cdef:
		int n_Par = vPar.shape[0]
		int m_mask = vMask.shape[0]
		int m = mY.shape[0]
		int N = mY.shape[1]
		kf kfpack
	return kfpack.callLBFGS(&vPar[0],&vMask[0],&vPams[0],&mY[0,0],
		n_Par,m_mask,r,m,N,&fx,MaxIter,delta,epsilon,ftol,gtol,mhess,iPrint)






