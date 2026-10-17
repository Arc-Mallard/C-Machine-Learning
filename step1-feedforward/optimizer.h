#include "../matrix/matrix.h"
#include <math.h>
#include <stdbool.h>

#define EPSILON 10e-7f

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamParams;

typedef struct {
	AdamParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
} AdamState;

void stateAssign(Matrix* s, Matrix m);

AdamState adamInit(Network net, float learningRate, float weightDecay);

void adamFree(AdamState* state, Network net);

void adamwStep(AdamParams p, Matrix f, Matrix s, Matrix g, Matrix* w1, Matrix w2, float frstPowf, float scndPowf, int w);

Network adamwOptimize(AdamState* state, Network net, Matrix wghtGrad[], Matrix biasGrad[]);
