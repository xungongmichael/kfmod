#include <lbfgs.h>
#include "kfpack.h"

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
	int iPrint;
}OptData;

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    OptData *optdata = (OptData *)instance;
    
    kf kfpack;
    
    //Calcualte Gradient Using Intel parallel
    kfpack.NumGradient(g, x, optdata->vMask, optdata->vPams, optdata->data, optdata->m_mask, optdata->n_par, optdata->r, optdata->m, optdata->N);

    //Evaluate likelihood function value
    double adFunc = kfpack.MICevallik(x, optdata->vMask, optdata->vPams, optdata->data, optdata->m_mask, optdata->r,optdata->m,optdata->N); 

    return adFunc;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{	
	OptData *optdata = (OptData*)instance;
    if(k%optdata->iPrint==0)
    {
    printf("Iteration %d:\n", k);
    printf("  fx = %f, xnorm = %f, gnorm = %f\n", fx, xnorm, gnorm);
    }

    return 0;
}

int kf::callLBFGS(double *vPar, 
	int *vMask, double *vPams, double *data, 
	int n_par, int m_mask,
	int r, int m, int N,
	double *fx,
	int MaxIter, double delta, double epsilon, double ftol, double gtol, int mhess,
	int iPrint
	)
{

    int ret;
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.delta = delta;
    param.epsilon = epsilon;
    param.ftol = ftol;
    param.wolfe = 0.1;
    param.gtol = gtol;
    param.m = mhess;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = MaxIter;

    OptData optdata;
    optdata.vMask = vMask;
    optdata.vPams = vPams;
    optdata.data  = data;
    optdata.n_par = n_par;
    optdata.m_mask = m_mask;
    optdata.r = r;
    optdata.m =m;
    optdata.N = N;
    optdata.iPrint = iPrint;

    ret = lbfgs(n_par, vPar, fx, evaluate, progress, (void *)&optdata, &param);
    return ret;
}

