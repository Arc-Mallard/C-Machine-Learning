#include "../matrix/matrix.h"
#include <math.h>
#include <stdbool.h>

#define EPSILON 10e-7f

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamwParams;

typedef struct {
	AdamwParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
} AdamwState;

void stateAssign(Matrix* s, Matrix m);

AdamwState adamwInit(Network net, float learningRate, float weightDecay);

void adamwFree(AdamwState* state, Network net);

void adamwStep(AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix* w1, Matrix w2, float frstPowf, float scndPowf, int w);

Network adamwOptimize(AdamwState* state, Network net, Matrix wghtGrad[], Matrix biasGrad[]);
