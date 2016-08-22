from scipy.linalg import eigh
import numpy as np
import warnings
__all__ = [
    "isPSD",
    "_input_checker",
    "_fortran_checker"
]

def isPSD(A, tol=1e-8):
		E,V = eigh(A)
		return np.all(E > -tol)


def _input_dim_checker(mY , vC = None, mZ = None, mH = None, 
		vD = None, mT = None, mQ = None, a1 = None, p1 = None,
		n_r = None):
	if not isinstance(mY, np.ndarray):
		raise TypeError("Input observations mY is not numpy object!")

	if mY.shape[0] > mY.shape[1]:
		warnings.warn("Input observations mY seems have more dimensions than sample length, make sure mY is a column matrix", RuntimeWarning)

	if vC is not None:
		if not isinstance(vC, np.ndarray):
			raise TypeError("Input observations offset vC is not numpy object!")
		if vC.shape[0] != mY.shape[0]:
			raise TypeError("row(vC)!=row(mY)")

	if mZ is not None:
		if not isinstance(mZ, np.ndarray):
			raise TypeError("Input observations transistion mZ is not numpy object!")
		if mZ.shape[0] != mY.shape[0]:
			raise TypeError("rows(mZ)!=rows(mY)!")
		if mZ.shape[1] != n_r:
			raise TypeError("cols(mZ)!= n_r!")

	if mH is not None:
		if not isinstance(mH, np.ndarray):
			raise TypeError("Input observations covariance mH is not numpy object!")
		if mH.shape[0] != mH.shape[1]:
			raise TypeError("mH must be a square matrix")
		if mH.shape[1] != mY.shape[0]:
			raise TypeError("row(mH) != row(mY)!")
		if not (mH.transpose() == mH).all():
			raise ValueError("mH is not symmetric!")
		if not isPSD(mH):
			#raise ValueError, "mH is not positive definite"
			pass

	if vD is not None:
		if not isinstance(vD, np.ndarray):
			raise TypeError("Input state offset vD is not numpy object!")
		if vD.shape[0] != n_r:
			raise ValueError("row(vD) != n_r")

	if mT is not None:
		if not isinstance(mT, np.ndarray):
			raise TypeError("Input state transistion mT is not numpy object!")
		if mT.shape[1] != n_r:
			raise TypeError("col(mT) != n_r")

	if mQ is not None:
		if not isinstance(mQ, np.ndarray):
			raise TypeError("Input state variance mQ is not numpy object!")	
		if mQ.shape[0] != mQ.shape[1]:
			raise TypeError("mQ must be a square matrix")
		if mQ.shape[1] != n_r:
			raise TypeError("row(mQ) != n_r)!")
		#if not (mQ.transpose() == mQ).all():
		#	raise ValueError, "mQ is not symmetric!"		
		if not isPSD(mQ):
			raise ValueError("mQ is not positve definite!")

	if a1 is not None:
		if not isinstance(a1, np.ndarray):
			raise TypeError("Input state offset vD is not numpy object!")
		if a1.shape[0] != n_r:
			raise ValueError("row(a1) != n_r")

	if p1 is not None:
		if not isinstance(p1, np.ndarray):
			raise TypeError("Input state transistion mT is not numpy object!")
		if p1.shape[0] != p1.shape[1]:
			raise TypeError("p1 must be a square matrix!")
		if not isPSD(p1):
			raise ValueError("p1 must be positve definite matrix")
		if p1.shape[1] != n_r:
			raise TypeError("col(p1) != n_r")

	# if np.all(mZ,vD,mT,mQ)==None and n_r==None:
	# 	raise TypeError, "mZ,vD,mT,mQ are all none, dimensions of states cannot be deduced"

def _fortran_checker(mY , vC = None, mZ = None, mH = None, 
		vD = None, mT = None, mQ = None, a1 = None, p1 = None):
		
		if not mY.flags.f_contiguous:
			mY = np.asfortranarray(mY)

		if vC is not None:
			if not vC.flags.f_contiguous:
				vC = np.asfortranarray(vC)

		if mZ is not None:
			if not mZ.flags.f_contiguous:
				mZ = np.asfortranarray(mZ)

		if mH is not None:		
			if not mH.flags.f_contiguous:
				mH = np.asfortranarray(mH)

		if vD is not None:
			if not vD.flags.f_contiguous:
				vD = np.asfortranarray(vD)

		if mT is not None:
			if not mT.flags.f_contiguous:
				mT = np.asfortranarray(mT)

		if mQ is not None:
			if not mQ.flags.f_contiguous:
				mQ = np.asfortranarray(mQ)

		if a1 is not None:
			if not a1.flags.f_contiguous:
				a1 = np.asfortranarray(a1)

		if p1 is not None:
			if not p1.flags.f_contiguous:
				p1 = np.asfortranarray(p1)


		return mY,vC,mZ,mH,vD,mT,mQ,a1,p1
							