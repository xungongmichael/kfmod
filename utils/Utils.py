import numpy as np 
import matplotlib.pylab as plt

def LDL(A):
    n = A.shape[1]
    L = np.array(np.eye(n))
    D = np.zeros((n, 1))
    for i in range(n):
        D[i] = A[i, i] - np.dot(L[i, 0:i] ** 2, D[0:i])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, 0:i] * L[i, 0:i], D[0:i])) / D[i]
    return [L, np.diag(D.reshape(D.shape[0]))]


def GenrSam(r,m,N,seed=123,ret = 'true', twist_level=0.01):
	np.random.seed(seed)
	mHdiag = np.exp(np.random.normal(-3,0.1,m))
	mH = np.diag(mHdiag)
	mZ = np.random.normal(1,1,[m,r])
	vC = np.random.normal(0,0.001,m)

	x0 = np.random.normal(0,1,[r,1])
	vD = np.random.normal(0,0.001,r)
	mT = np.random.normal(0.0,0.01,[r,r])
	mT[np.diag_indices(r)] = np.random.normal(0.9,0.01,r)
	mQ = np.random.normal(0.000051,0.00001,[r,r])
	mQ[np.diag_indices(r)] = np.exp(np.random.normal(-5,0.1,r))
	mQ = (mQ+mQ.T)/2
	mX = np.zeros([r,N])
	mY = np.zeros([m,N])
	mX[:,0] = x0.ravel()
	mY[:,0] = vC + np.dot(mZ,x0).ravel() + np.random.multivariate_normal(np.zeros(m),mH)

	for i in range(1,N):
		epsQ = np.random.multivariate_normal(np.zeros(r), mQ)
		epsH = np.random.multivariate_normal(np.zeros(m), mH)
		mX[:,i] = vD + np.dot(mT,mX[:,i-1]) + epsQ
		mY[:,i] = vC + np.dot(mZ,mX[:,i]).ravel() + epsH
	
	if ret == 'true':
		return vC.reshape([m,1]),mZ,mH,vD.reshape([r,1]),mT,mQ,mY,mX
	if ret == 'twist':
		vC += np.random.normal(0,0.001*twist_level,m)
		mZ += np.random.normal(0,1*twist_level,[m,r])
		mH[np.diag_indices(m)] += np.random.normal(0.0001,0.0001*twist_level,m)
		vD += np.random.normal(0,0.001*twist_level,r)
		mT += np.random.normal(0,0.01*twist_level,[r,r])
		mQ += np.random.normal(0.0,0.00001*twist_level,[r,r])
		mQ = (mQ+mQ.T)/2
		return vC.reshape([m,1]),mZ,mH,vD.reshape([r,1]),mT,mQ,mY,mX

def LBFGSretcode(retcode):
	code = {0:"L-BFGS reaches convergence.",
	1:"LBFGS_CONVERGENCE",
	2:"The initial variables already minimize the objective function.",
	-1024:"Unknown error.",
	-1023:"Logic error.",
	-1022:"Insufficient memory.",
	-1021:"The minimization process has been canceled.",
	-1020:"Invalid number of variables specified.",
	-1019:"Invalid number of variables (for SSE) specified.",
	-1018:"The array x must be aligned to 16 (for SSE).",
	-1017:"Invalid parameter lbfgs_parameter_t::epsilon specified.",
	-1016:"Invalid parameter lbfgs_parameter_t::past specified.",
	-1015:"Invalid parameter lbfgs_parameter_t::delta specified.",
	-1014:"Invalid parameter lbfgs_parameter_t::linesearch specified.",
	-1013:"Invalid parameter lbfgs_parameter_t::max_step specified.",
	-1012:"Invalid parameter lbfgs_parameter_t::max_step specified.",
	-1011:"Invalid parameter lbfgs_parameter_t::ftol specified.",
	-1010:"Invalid parameter lbfgs_parameter_t::wolfe specified.",
	-1009:"Invalid parameter lbfgs_parameter_t::gtol specified.",
	-1008:"Invalid parameter lbfgs_parameter_t::xtol specified.",
	-1007:"Invalid parameter lbfgs_parameter_t::max_linesearch specified.",
	-1006:"Invalid parameter lbfgs_parameter_t::orthantwise_c specified.",
	-1005:"Invalid parameter lbfgs_parameter_t::orthantwise_start specified.",
	-1004:"Invalid parameter lbfgs_parameter_t::orthantwise_end specified.",
	-1003:"The line-search step went out of the interval of uncertainty.",
	-1002:"A logic error occurred; alternatively, the interval of uncertainty became too small.",
	-1001:"A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions.",
	-1000:"The line-search step became smaller than lbfgs_parameter_t::min_step.",
	-999:"The line-search step became larger than lbfgs_parameter_t::max_step.",
	-998:"The line-search routine reaches the maximum number of evaluations.",
	-997:"The algorithm routine reaches the maximum number of iterations.",
	-996:"Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol.",
	-995:"A logic error (negative line-search step) occurred.",
	-994:"The current search direction increases the objective function value."}

	return code[retcode]


def vec(a):
	return a.reshape(a.size,1,order='F')

def _coef_decomp(M,cons):
	M_masked = ma.array(M,mask=cons)
	cons = vec(cons).astype(dtype=np.bool)
	vM = vec(M)

	m = vec(M_masked.flatten('F').compressed())
	f = np.zeros_like(vM)
	f[np.nonzero(cons)] = vM[cons.flatten()].flatten()

	D = np.zeros([M.size,m.size])
	D[cons.flatten()==0,:] = np.eye(m.size)	
	assert (vM == (f + np.dot(D,m))).all(), \
			"Coefficient matrix decompose fails"

	f = np.asfortranarray(f)
	D = np.asfortranarray(D)
	m = np.asfortranarray(m)
	return (f, D, m)