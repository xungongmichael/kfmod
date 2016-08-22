#include "kfpack.h"
#include <omp.h>
#include <mkl.h>
#define EIGEN_NO_DEBUG
#define EIGEN_USE_MKL_ALL
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <Eigen/Dense>

using namespace Eigen;

//NOTE: Need to solve the dynamic allocation problem
void _Moment_core(double *p_a1, double *p_p1, const VectorXd &vC, 
	const MatrixXd &mH, double *p_vD , double *p_mT, double *p_mQ, 
	const MatrixXd &mY, double *p_mA_filter, double *p_mP_filter, 
	double *p_mA_predictor, double *p_mP_predictor, const int r, const int N){

	Map<VectorXd> a1(p_a1,r);
	Map<Matrix<double,Dynamic,Dynamic>> p1(p_p1,r,r);

	Map<VectorXd> vD(p_vD,r);
	Map<Matrix<double,Dynamic,Dynamic>> mT(p_mT,r,r);
	Map<Matrix<double,Dynamic,Dynamic>> mQ(p_mQ,r,r);

	Map<Matrix<double,Dynamic,Dynamic>> mA_F(p_mA_filter,r,N);
	Map<Matrix<double,Dynamic,Dynamic>> mP_F(p_mP_filter,r,r*N);
	Map<Matrix<double,Dynamic,Dynamic>> mA_P(p_mA_predictor,r,N);
	Map<Matrix<double,Dynamic,Dynamic>> mP_P(p_mP_predictor,r,r*N);

	VectorXd at = a1;
	MatrixXd pt = p1;

	VectorXd v, att;
	MatrixXd ptt, F;
	LDLT<MatrixXd> invF(r);
	for (int i = 0; i < N; ++i)
	{
		//Error decomposition
		v.noalias() = mY.col(i) - at - vC;
		F.noalias() = pt + mH;

		//Kalman Filter
		invF.compute(F);
		att.noalias() = at + pt*invF.solve(v);
		ptt.noalias() = pt - pt*invF.solve(pt);
		mA_F.col(i) = att;
		mP_F.block(0,i*r,r,r) = ptt;

		//Kalman Predictor
		at.noalias() = mT*att + vD;
		pt.noalias() = mT*ptt*mT.transpose() + mQ;
		mA_P.col(i) = at;
		mP_P.block(0,i*r,r,r) = pt;		
	}
}


void kf::Moment(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, 
	double *p_mH, double *p_vD,double *p_mT, double *p_mQ, double *p_mY, 
	double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, 
	double *p_mP_predictor, int r, int m, int N)
{
	Map<VectorXd> vC(p_vC,m);
	Map<Matrix<double,Dynamic,Dynamic>>mZ(p_mZ,m,r);
	Map<Matrix<double,Dynamic,Dynamic>>mH(p_mH,m,m);
	Map<Matrix<double,Dynamic,Dynamic>>mY(p_mY,m,N);

	ArrayXd vinvH(m);
	MatrixXd invH(m,m);
	vinvH = 1/mH.diagonal().array();
	invH = vinvH.matrix().asDiagonal();
	
	MatrixXd ZHZ, mC;
	ZHZ.noalias() = mZ.transpose()*invH*mZ;
	mC.noalias() = ZHZ.inverse();

	MatrixXd mAhat(r,N), mChat(r,1), mProj(r,m);
	mProj.noalias() = mC*mZ.transpose()*invH;
	mAhat.noalias() = mProj*mY;
	mChat.noalias() = mProj*vC;

	_Moment_core(p_a1,p_p1,mChat,mC,p_vD,p_mT,p_mQ,
		mAhat,p_mA_filter,p_mP_filter,p_mA_predictor,p_mP_predictor,r,N);
}


void _Smooth_core(double *p_a1, double *p_p1, const VectorXd &vC, 
	const MatrixXd &mH, double *p_vD , double *p_mT, double *p_mQ, 
	const MatrixXd &mY, double *p_mA_filter, double *p_mP_filter, 
	double *p_mA_predictor, double *p_mP_predictor, 
	double *p_mA_smoother, double *p_mP_smoother,
	const int r, const int N){

	Map<VectorXd> a1(p_a1,r);
	Map<Matrix<double,Dynamic,Dynamic>> p1(p_p1,r,r);

	Map<VectorXd> vD(p_vD,r);
	Map<Matrix<double,Dynamic,Dynamic>> mT(p_mT,r,r);
	Map<Matrix<double,Dynamic,Dynamic>> mQ(p_mQ,r,r);

	Map<Matrix<double,Dynamic,Dynamic>> mA_F(p_mA_filter,r,N);
	Map<Matrix<double,Dynamic,Dynamic>> mP_F(p_mP_filter,r,r*N);
	Map<Matrix<double,Dynamic,Dynamic>> mA_P(p_mA_predictor,r,N+1);
	Map<Matrix<double,Dynamic,Dynamic>> mP_P(p_mP_predictor,r,r*(N+1));
	Map<Matrix<double,Dynamic,Dynamic>> mA_S(p_mA_smoother,r,N);
	Map<Matrix<double,Dynamic,Dynamic>> mP_S(p_mP_smoother,r,r*N);

	Matrix<double,Dynamic,Dynamic> mV(r,N);
	Matrix<double,Dynamic,Dynamic> minvF(r,N*r);
	Matrix<double,Dynamic,Dynamic> mL(r,N*r);

	MatrixXd F, K;
	//Intialize Kalman Filter
	mA_P.col(0) = a1;
	mP_P.block(0,0,r,r) = p1;

	for (int i = 0; i < N; ++i)
	{
		//Error decomposition
		mV.col(i) = mY.col(i) - mA_P.col(i) - vC;
		F = mP_P.block(0,i*r,r,r) + mH;
		minvF.block(0,i*r,r,r) = F.inverse();
		K = mT*mP_P.block(0,i*r,r,r)*minvF.block(0,i*r,r,r);
		mL.block(0,i*r,r,r) = mT - K;

		//Kalman Filter
		mA_F.col(i).noalias() = mA_P.col(i) + 
				mP_P.block(0,i*r,r,r)*minvF.block(0,i*r,r,r)*mV.col(i);
		mP_F.block(0,i*r,r,r).noalias() = mP_P.block(0,i*r,r,r) - 
				mP_P.block(0,i*r,r,r)*minvF.block(0,i*r,r,r)*mP_P.block(0,i*r,r,r);

		//Kalman Predictor
		mA_P.col(i+1) = mT*mA_F.col(i) + vD;
		mP_P.block(0,(i+1)*r,r,r) = mT*mP_F.block(0,i*r,r,r)*mT.transpose() + mQ;		
	}

	//Start Smoother
	VectorXd rt = VectorXd::Zero(r);
	MatrixXd Nt = MatrixXd::Zero(r,r);
	for (int i = N-1; i >= 0; --i)
	{
		rt = minvF.block(0,i*r,r,r)*mV.col(i) + mL.block(0,i*r,r,r).transpose()*rt;
		mA_S.col(i) = mA_P.col(i) + mP_P.block(0,i*r,r,r)*rt;
		Nt = minvF.block(0,i*r,r,r) + mL.block(0,i*r,r,r).transpose()*Nt*mL.block(0,i*r,r,r);
		mP_S.block(0,i*r,r,r) = mP_P.block(0,i*r,r,r) - mP_P.block(0,i*r,r,r)*Nt*mP_P.block(0,i*r,r,r);

	}


}

void kf::Smooth(double *p_a1, double *p_p1, double *p_vC, double *p_mZ, 
	double *p_mH, double *p_vD,double *p_mT, double *p_mQ, double *p_mY, 
	double *p_mA_filter, double *p_mP_filter, double *p_mA_predictor, 
	double *p_mP_predictor, double *p_mA_smoother, double *p_mP_smoother,
	int r, int m, int N)
{
	Map<VectorXd> vC(p_vC,m);
	Map<Matrix<double,Dynamic,Dynamic>>mZ(p_mZ,m,r);
	Map<Matrix<double,Dynamic,Dynamic>>mH(p_mH,m,m);
	Map<Matrix<double,Dynamic,Dynamic>>mY(p_mY,m,N);

	ArrayXd vinvH(m);
	MatrixXd invH(m,m);
	vinvH = 1/mH.diagonal().array();
	invH = vinvH.matrix().asDiagonal();
	
	MatrixXd ZHZ = mZ.transpose()*invH*mZ;
	MatrixXd mC = ZHZ.inverse();

	MatrixXd mAhat(r,N), mChat(r,1), mProj(r,m);
	mProj = mC*mZ.transpose()*invH;
	mAhat = mProj*mY;
	mChat = mProj*vC;

	_Smooth_core(p_a1,p_p1,mChat,mC,p_vD,p_mT,p_mQ,
		mAhat,p_mA_filter,p_mP_filter,p_mA_predictor,p_mP_predictor,p_mA_smoother,p_mP_smoother,r,N);
}
