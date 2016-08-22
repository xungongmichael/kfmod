#include <cmath>
#include <omp.h>
#include <mkl.h>
#include <cstring>
#define EIGEN_NO_DEBUG
#define EIGEN_USE_MKL_ALL
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "kfpack.h"
#define PI 3.14159265358979

using namespace Eigen;
using namespace std;

typedef Matrix<double,5,5> Matrix5d;
typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,5,1> Vector5d;
typedef Matrix<double,6,1> Vector6d;


/***********************
Size 1 Implementation
***********************/
double KFLik1d(const double &a1, const double &p1, const double &vC, const double &mH, const double &vD, const double &mT, const double &mQ, const double *mY, const int r, const int m, const int N){

	double a = a1, v;
	double P = p1, F,invF, K, P_update;
	double logdetFsum = 0,invFVVsum = 0, detF;
	
	for (int i = 0; i < N; ++i)
	{
		v = mY[i] - a - vC;
		F = P + mH;

		detF = log(F);
		logdetFsum += detF;

		invF = 1/F;
		invFVVsum += v*invF*v;

		K = mT*P*invF;
		a = vD + mT*a + K*v;
		P = mT*P*mT + mQ - K*F*K;
	}

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size 2 Implementation
***********************/
double KFLik2d(const Vector2d &a1, const Matrix2d &p1, const Vector2d &vC, const Matrix2d &mH, const Vector2d &vD, const Matrix2d &mT, const Matrix2d &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	Vector2d a = a1, v;
	Matrix2d P = p1, F,invF, K, P_update;
	double logdetFsum = 0,invFVVsum = 0, detF;
	
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

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size 3 Implementation
***********************/
double KFLik3d(const Vector3d &a1, const Matrix3d &p1, const Vector3d &vC, const Matrix3d &mH, const Vector3d &vD, const Matrix3d &mT, const Matrix3d &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	Vector3d a = a1, v;
	Matrix3d P = p1, F,invF, K, P_update;
	double logdetFsum = 0,invFVVsum = 0, detF;
	
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

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size 4 Implementation
***********************/
double KFLik4d(const Vector4d &a1, const Matrix4d &p1, const Vector4d &vC, const Matrix4d &mH, const Vector4d &vD, const Matrix4d &mT, const Matrix4d &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	Vector4d a = a1, v;
	Matrix4d P = p1, F,invF, K, P_update;
	double logdetFsum = 0,invFVVsum = 0, detF;	

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

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size 5 Implementation
***********************/
double KFLik5d(const Vector5d &a1, const Matrix5d &p1, const Vector5d &vC, const Matrix5d &mH, const Vector5d &vD, const Matrix5d &mT, const Matrix5d &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	Vector5d a = a1, v;
	Matrix5d P = p1, F,invF, K, P_update, I = MatrixXd::Identity(r,r);
	double logdetFsum = 0,invFVVsum = 0, detF;
	
	for (int i = 0; i < N; ++i)
	{
		v = mY.col(i) - a - vC;
		F = P + mH;

		detF = log(F.determinant());
		logdetFsum += detF;

		invF = F.ldlt().solve(I);
		invFVVsum += v.transpose()*invF*v;

		K = mT*P*invF;
		a = vD + mT*a + K*v;
		P = mT*P*mT.transpose() + mQ - K*F*K.transpose();
	}

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size 6 Implementation
***********************/
double KFLik6d(const Vector6d &a1, const Matrix6d &p1, const Vector6d &vC, const Matrix6d &mH, const Vector6d &vD, const Matrix6d &mT, const Matrix6d &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	Vector6d a = a1, v;
	Matrix6d P = p1, F,invF, K, P_update, I = MatrixXd::Identity(r,r);
	double logdetFsum = 0,invFVVsum = 0, detF;
	
		for (int i = 0; i < N; ++i)
		{
			v = mY.col(i) - a - vC;
			F = P + mH;

			detF = log(F.determinant());
			logdetFsum += detF;

			invF = F.ldlt().solve(I);
			invFVVsum += v.transpose()*invF*v;

			K = mT*P*invF;
			a = vD + mT*a + K*v;
			P = mT*P*mT.transpose() + mQ - K*F*K.transpose();
		}

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


/***********************
Size General Implementation
***********************/
double KFLikXd(const VectorXd &a1, const MatrixXd &p1, const VectorXd &vC, const MatrixXd &mH, const VectorXd &vD, const MatrixXd &mT, const MatrixXd &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	VectorXd a = a1, v;
	MatrixXd P = p1, F,invF, K, P_update, I = MatrixXd::Identity(r,r);
	double logdetFsum = 0,invFVVsum = 0, detF;

		for (int i = 0; i < N; ++i)
		{
			v = mY.col(i) - a - vC;
			F = P + mH;

			detF = log(F.determinant());
			logdetFsum += detF;

			invF = F.ldlt().solve(I);
			invFVVsum += v.transpose()*invF*v;

			K = mT*P*invF;
			a = vD + mT*a + K*v;
			P = mT*P*mT.transpose() + mQ - K*F*K.transpose();
		}

	double LikVal = -0.5*N*r*log(2*PI) - 0.5*logdetFsum - 0.5*invFVVsum;
	return LikVal;
}


double DFMLik(const VectorXd &a1, const MatrixXd &p1, const VectorXd &vC, const MatrixXd &mZ, const MatrixXd &mH, const VectorXd &vD, const MatrixXd &mT,  const MatrixXd &mQ, const MatrixXd &mY, const int r, const int m, const int N){

	ArrayXd vinvH = 1/mH.diagonal().array();
	MatrixXd invH = vinvH.matrix().asDiagonal();
	
	MatrixXd ZHZ = mZ.transpose()*invH*mZ;
	MatrixXd mC = ZHZ.inverse();
	//invF = F.ldlt().solve(Matrix3d::Identity());
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
		case 1: 
			core_lik = KFLik1d(a1(0),p1(0),vChat(0),mC(0),vD(0),mT(0),mQ(0),
				mAhat.data(),r,m,N);break;
		case 2: 
			core_lik = KFLik2d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
		case 3: 
			core_lik = KFLik3d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
		case 4: 
			core_lik = KFLik4d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
		case 5: 
			core_lik = KFLik5d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
		case 6: 
			core_lik = KFLik6d(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
		default: 
			core_lik = KFLikXd(a1,p1,vChat,mC,vD,mT,mQ,mAhat,r,m,N);break;
	}

	double NegLoglik = 0.5*N*(m-r)*log(2*PI) - 
					   core_lik + 
					   0.5*vFv + 
					   0.5*N*log(detH/detC);		   
	return NegLoglik;
}


double kf::MICevallik(
	const double *vPar, 
	int *vMask, 
	double *vPams, 
	double *p_mY, 
	int m_mask, int r, int m, int N)
{	
	Map<Matrix<double,Dynamic,Dynamic>>mY(p_mY,m,N);
	
	int j = 0;
	double newpar[m_mask];
	for (int i = 0; i < m_mask; ++i)
	{
		if(vMask[i]){
			newpar[i] = vPams[i];
		}else{
			newpar[i] = vPar[j];
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

	Map<VectorXd>vC   (&newpar[0],m);
	Map<MatrixXd>mZ   (&newpar[idx_mZ],m,r);
	Map<MatrixXd>mH   (&newpar[idx_mH],m,m);
	Map<VectorXd>vD   (&newpar[idx_vD],r);
	Map<MatrixXd>mT   (&newpar[idx_mT],r,r);
	Map<MatrixXd>mQ_L (&newpar[idx_mQ_L],r,r);
	Map<MatrixXd>mQ_D (&newpar[idx_mQ_D],r,r);

	mH.diagonal() = mH.diagonal().array().exp();
	mQ_D.diagonal() = mQ_D.diagonal().array().exp();
	MatrixXd mQ = mQ_L*mQ_D*mQ_L.transpose();

	VectorXd va1 = VectorXd::Zero(r);
	MatrixXd mP1 = MatrixXd::Identity(r,r);

    double LikVal = DFMLik(va1,mP1,vC,mZ,mH,vD,mT,mQ,mY,r,m,N);
	return LikVal;
}



double kf::_EvalFunc(int i, double *avP,  int *vMask,  double *vPams, 
	 double *mY, int r,int m, int N, int m_mask){

    double p,fp,fm,jac,h;
    double jac_eps = 1E-8;

    p = avP[i];
    // h = abs(p) * jac_eps;
    h = jac_eps;
    // if (p == 0.0) h = jac_eps;

    avP[i] = p + h;
    fp = kf::MICevallik(avP, vMask, vPams, mY, m_mask, r, m, N);

    avP[i] = p - h;
    fm = kf::MICevallik(avP, vMask, vPams, mY, m_mask, r, m, N);

    avP[i] = p;

    jac = (fp-fm)/(2*h);
    return jac;
}


int kf::NumGradient(
	double *vScore, 
	const double *vPar, 
	int *vMask, 
	double *vPams, 
	double *mY, 
	int m_mask, int num,
	int r, int m, int N)
{
 
    int  i;
    double vP[num];
    memcpy(&vP[0],vPar,sizeof(double)*num);

    #pragma omp parallel for firstprivate(vP) shared(vScore, vMask, vPams, mY, r, m, N, m_mask) private(i) schedule(dynamic)
    for (i = 0; i < num; ++i)
    {
        vScore[i] = _EvalFunc(i,vP, vMask, vPams, mY, r, m, N, m_mask);
    }

    return 1;
}


/*  VectorXd va1 = (MatrixXd::Identity(r,r)-mT).inverse()*vD;
    MatrixXd mP1(r*r,1);
    Map<VectorXd>vQ(mQ.data(),r*r);
    mP1 = (MatrixXd::Identity(r*r,r*r)-kroneckerProduct(mT,mT)).inverse()*vQ;
    mP1.resize(r,r);*/
    //Map<MatrixXd>mP1(vP1.data(),r,r);