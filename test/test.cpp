#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <mkl.h>
#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <Eigen/Dense>
#include <include/cppoptlib/meta.h>
#include <include/cppoptlib/problem.h>
#include <include/cppoptlib/solver/bfgssolver.h>


#define PI 3.14159265358979
#define MAX_LINE_LENGTH 102400
#define MAX_CSV_ELEMENTS 1000000


using namespace Eigen;
using namespace std;
typedef struct {
	int *vMask; 
	double *vPams; 
	double *data; 
	int n_par; 
	int m_mask;
	int r;
	int m;
	int N;
	int steady;
	int iPrint;
}OptData;

void ReadData(char *filename, double *CSVArray){
    char line[MAX_LINE_LENGTH] = {0};
    int varCount = 0;
    FILE *csvFile = fopen(filename, "r");
    if (csvFile)
    {   
        char *token = 0;
        while (fgets(line, MAX_LINE_LENGTH, csvFile)) 
        {
            token = strtok(&line[0], ",");
            while (token)
            {
                CSVArray[varCount] = atof(token);
                token = strtok(NULL, ",");
                varCount++;
            }
        }
        fclose(csvFile);
    }
}




/***********************
Size 3 Implementation
***********************/
double KFLik3d(const Vector3d &a1, const Matrix3d &p1, const Vector3d &vC, const Matrix3d &mH, const Vector3d &vD, const Matrix3d &mT, const Matrix3d &mQ, const MatrixXd &mY, const int r, const int m, const int N, const int steady){

	Vector3d a = a1, v;
	Matrix3d P = p1, F,invF, K, P_update;
	double logdetFsum = 0,invFVVsum = 0, detF;
	
	switch(steady){
		case 0:
			for (int i = 0; i < N; ++i)
			{
				v = mY.col(i) - a - vC;
				F = P + mH;

				detF = log(F.determinant());
				logdetFsum += detF;

				invF = F.inverse();
				invFVVsum += v.transpose()*invF*v;

				K = mT*P*invF;
				a = vD + mT*a + K*v;
				P = mT*P*mT.transpose() + mQ - K*F*K.transpose();
			}
		break;

		case 1:
			int i = 0;
			for (i = 0; i < N; ++i)
			{
				v = mY.col(i) - a - vC;
				F = P + mH;

				detF = log(F.determinant());
				logdetFsum += detF;

				invF = F.inverse();
				invFVVsum += v.transpose()*invF*v;

				K = mT*P*invF;
				a = vD + mT*a + K*v;
				P_update = mT*P*mT.transpose() + mQ - K*F*K.transpose();

				if ((P-P_update).array().abs().maxCoeff()<1E-16) break;
				else P = P_update;
			}

			for (int j = i; j < N; ++j)
			{
				v = mY.col(j) - a - vC;
				invFVVsum += v.transpose()*invF*v;
				a = vD + mT*a + K*v;
			}
			logdetFsum += (N-i-1)*detF;
		break;
	}

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}



double DFMLik(const VectorXd &a1, const MatrixXd &p1, const VectorXd &vC, const MatrixXd &mZ, const MatrixXd &mH, const VectorXd &vD, const MatrixXd &mT,  const MatrixXd &mQ, const MatrixXd &mY, const int r, const int m, const int N, const int steady){

	ArrayXd vinvH = 1/mH.diagonal().array();
	MatrixXd invH = vinvH.matrix().asDiagonal();
	
	MatrixXd ZHZ = mZ.transpose()*invH*mZ;
	MatrixXd mC = ZHZ.inverse();

	MatrixXd mAhat(r,N), vChat(r,1), mProj(r,m);
	mProj = mC*mZ.transpose()*invH;
	mAhat = mProj*mY;
	vChat = mProj*vC;

	MatrixXd mE = mY-mZ*mAhat;

	double detC = mC.determinant();
	double detH = mH.determinant();
	double vFv = ((invH.transpose()*mE).array()*mE.array()).sum();

	double core_lik;
	switch(r){
		case 2: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
		case 3: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
		case 4: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
		case 5: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
		case 6: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
		default: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N,steady);break;
	}
	

	double NegLoglik = 0.5*N*(m-r)*log(2*PI) - 
					   core_lik + 
					   0.5*vFv + 
					   0.5*N*log(detH/detC);		   
	return NegLoglik;
}


double MICevallik(const VectorXd &vPar, const VectorXi &vMask, const VectorXd &vPams, 
	const MatrixXd &mY, int r,int m, int N, int steady)
{	
	int m_mask = vMask.rows();
	
	int j = 0;
	//double *newpar = new double [m_mask];
	double newpar[m_mask];
	for (int i = 0; i < m_mask; ++i)
	{
		if(vMask(i)){
			newpar[i] = vPams(i);
		}else{
			newpar[i] = vPar(j);
			j+=1;
		}
	}
	
	int idx_vC = 0;
	int idx_mZ = idx_vC + m;
	int idx_mH = idx_mZ + r*m;
	int idx_vD = idx_mH + m*m;
	int idx_mT = idx_vD + r*1;
	int idx_mQ_L = idx_mT + r*r;
	int idx_mQ_D = idx_mQ_L + r*r;

	Map<VectorXd>vC(&newpar[0],m);
	Map<MatrixXd>mZ(&newpar[idx_mZ],m,r);
	Map<MatrixXd>mH(&newpar[idx_mH],m,m);
	Map<VectorXd>vD(&newpar[idx_vD],r);
	Map<MatrixXd>mT(&newpar[idx_mT],r,r);
	Map<MatrixXd>mQ_L(&newpar[idx_mQ_L],r,r);
	Map<MatrixXd>mQ_D(&newpar[idx_mQ_D],r,r);

	mH.diagonal() = mH.diagonal().array().exp();
	mQ_D.diagonal() = mQ_D.diagonal().array().exp();
	MatrixXd mQ = mQ_L*mQ_D*mQ_L.transpose();

    Vector3d va1;
    Matrix3d mP1;

    va1 << 0.00,0.00,0.00;
    mP1 = Matrix3d::Identity();

    double LikVal = DFMLik(va1,mP1,vC,mZ,mH,vD,mT,mQ,mY,r,m,N,steady);
	return LikVal;
}




double _EvalFunc(int i, VectorXd &avP, const VectorXi &vMask, const VectorXd &vPams, 
	const MatrixXd &mY, int r,int m, int N, int steady, double jac_eps){

    double p, fp, fm,jac;

    p = avP(i);

    avP(i) = p + jac_eps;
    fp = MICevallik(avP, vMask, vPams, mY, r, m, N, steady);

    avP(i) = p - jac_eps;
    fm = MICevallik(avP, vMask, vPams, mY, r, m, N, steady);

    avP(i) = p;

    jac = (fp-fm)/(2*jac_eps);
    return jac;
}


int NumGradient(VectorXd &vP, VectorXd &vScore, const VectorXi &vMask, const VectorXd &vPams, 
	const MatrixXd &mY, int r,int m, int N, int steady)
{
 
    int  i;
    int numPars = vP.rows();
    double jac_eps = 5E-6;
    VectorXd vP_cp = vP;

    #pragma omp parallel for firstprivate(vP_cp) shared(vScore, vMask, vPams, mY, r, m, N, steady, jac_eps) private(i) schedule(dynamic)
    for (i = 0; i < numPars; ++i)
    {
        vScore(i) = _EvalFunc(i,vP_cp, vMask, vPams, mY, r, m, N, steady, jac_eps);
    }

    return 1;
}


namespace cppoptlib {

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
class DynamicFactor : public Problem<double> {
    const MatrixXd mY;
    const Vector<int> vMask;
    const Vector<double> vPams;

  public:
    DynamicFactor(const MatrixXd &_mY, const Vector<int> &_vMask,
    	const Vector<double>&_vPams) : mY(_mY), vMask(_vMask), vPams(_vPams) {}

    double value(const Vector<double> &vPar) {
        return MICevallik(vPar, vMask, vPams, mY, 3,18, 3437, 1);
    }

    void gradient(const Vector<double> &vPar, Vector<double> &grad) {
    	VectorXd vP = vPar;
        NumGradient(vP, grad, vMask, vPams, mY, 3,18, 3437, 1);
    }
};

}









int main(){
	int vMask1[426] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
       1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
    double vPams1[426] = {
    	0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   1.25604e+00,   1.19944e+00,
         1.15921e+00,   1.04968e+00,   1.06227e+00,   1.05284e+00,
         1.00000e+00,   1.01585e+00,   1.00000e+00,   1.00000e+00,
         1.00555e+00,   1.00000e+00,   9.93419e-01,   9.81907e-01,
         9.65637e-01,   1.01530e+00,   9.74090e-01,   9.48282e-01,
        -3.45047e-01,   1.21077e+00,   2.54955e+00,  -9.33851e-01,
         3.55290e-01,   1.45494e+00,  -1.00000e+00,   9.41494e-02,
         1.00000e+00,  -1.00000e+00,   1.02021e-01,   1.00000e+00,
        -9.84871e-01,  -4.48429e-02,   7.30576e-01,  -9.44256e-01,
        -1.45871e-01,   5.09167e-01,   6.23320e+00,   8.24799e+00,
         8.75844e+00,   3.36300e+00,   3.41723e+00,   3.60977e+00,
         1.00000e+00,   6.86245e-01,   1.00000e+00,  -1.00000e+00,
        -1.17162e+00,  -1.00000e+00,  -3.08620e+00,  -3.02289e+00,
        -2.68097e+00,  -5.78291e+00,  -5.29328e+00,  -4.86490e+00,
         3.42890e-04,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   1.35938e-04,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   8.67387e-05,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   4.58223e-05,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         2.99045e-05,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   7.67072e-06,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   2.75999e-05,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   2.87291e-05,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         1.33326e-05,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   2.25833e-05,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   2.41241e-05,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   1.90947e-05,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         1.34666e-05,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   1.54701e-05,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   8.55239e-06,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   2.79918e-05,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         1.35978e-05,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   0.00000e+00,   0.00000e+00,   1.39979e-05,
         3.59059e-04,   4.26787e-04,   2.30472e-05,   1.00032e+00,
        -4.74492e-03,   3.77544e-04,   2.22389e-02,   9.75327e-01,
         1.22944e-03,  -7.52673e-02,   8.39176e-02,   9.86265e-01,
         1.00000e+00,  -4.23329e-01,   6.52692e-02,   0.00000e+00,
         1.00000e+00,  -1.83574e-02,   0.00000e+00,   0.00000e+00,
         1.00000e+00,   6.64075e-05,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   1.13816e-06,   0.00000e+00,   0.00000e+00,
         0.00000e+00,   4.46818e-08
     };
    double vPar[78] = {1.256 ,   1.1994,   1.1592,   1.0497,   1.0623,   1.0528,
         1.0159,   1.0055,   0.9934,   0.9819,   0.9656,   1.0153,
         0.9741,   0.9483,  -0.345 ,   1.2108,   2.5496,  -0.9339,
         0.3553,   1.4549,   0.0941,   0.102 ,  -0.9849,  -0.0448,
         0.7306,  -0.9443,  -0.1459,   0.5092,   6.2332,   8.248 ,
         8.7584,   3.363 ,   3.4172,   3.6098,   0.6862,  -1.1716,
        -3.0862,  -3.0229,  -2.681 ,  -5.7829,  -5.2933,  -4.8649,
        -7.9781,  -8.9033,  -9.3526,  -9.9907, -10.4175, -11.7781,
       -10.4977, -10.4576, -11.2253, -10.6983, -10.6323, -10.8661,
       -11.2153, -11.0766, -11.6693, -10.4836, -11.2056, -11.1766,
         0.0004,   0.0004,   0.    ,   1.0003,  -0.0047,   0.0004,
         0.0222,   0.9753,   0.0012,  -0.0753,   0.0839,   0.9863,
        -0.4233,   0.0653,  -0.0184,  -9.6197, -13.6861, -16.9237};

    char *dataname = "Vol_SPX_18.csv";
    const int r = 3;
    const int m = 18;
    const int N = 3437;
    double *data = (double *)mkl_calloc(N*(m+1), sizeof(double), 64);
    ReadData(dataname,data);

    OptData optdata;
    optdata.vMask = vMask1;
    optdata.vPams = vPams1;
    optdata.data  = data;
    optdata.n_par = 78;
    optdata.m_mask = 426;
    optdata.r = r;
    optdata.m =m;
    optdata.N = N;
    optdata.iPrint = 1;
    optdata.steady = 1;

    Map<const VectorXd>vP_map(vPar,78);
    //VectorXd vP = vP_map;
    VectorXd vScore(78);
    Map<VectorXi>vMask(optdata.vMask,optdata.m_mask);
    Map<VectorXd>vPams(optdata.vPams,optdata.m_mask);
    Map<Matrix<double,Dynamic,Dynamic>>mY(optdata.data,optdata.m,optdata.N);  

    //MatrixXd mdata = mY;
    cppoptlib::DynamicFactor f(mY, vMask,vPams);
    cppoptlib::Vector<double> vP = vP_map;
    std::cout << "start in   " << vP << std::endl;
    cppoptlib::BfgsSolver<double> solver;
    solver.minimize(f, vP);
    std::cout << "result     " << vP << std::endl;




}
