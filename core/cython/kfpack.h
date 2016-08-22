#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <lbfgs.h>
using namespace Eigen;
using namespace std;

class kf {
public:
	void Moment(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, double *p_mH, double *p_vD, double *p_mT, double *p_mQ, double *p_mY, double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, double *p_mP_predictor, int r, int m, int N);

	void Smooth(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, double *p_mH, double *p_vD,double *p_mT, double *p_mQ, double *p_mY, double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, double *p_mP_predictor, double *p_mA_smoother, double *p_mP_smoother,int r, int m, int N);

	double MICevallik(const double *vPar, int *vMask, double *vPams, double *p_mY, int m_mask, int r, int m, int N);

	double _EvalFunc(int i, double *avP,  int *vMask,  double *vPams, double *mY, int r,int m, int N, int m_mask);

	int NumGradient(double *vScore, const double *vPar, int *vMask, double *vPams, double *p_mY, int m_mask, int num, int r, int m, int N);

	int callLBFGS(double *vPar, int *vMask, double *vPams, double *data, int n_par, int m_mask, int r, int m, int N, double *fx, int MaxIter, double delta, double epsilon, double ftol, double gtol, int mhess, int iPrint);

};