#include "../matrix/matrix.h"
#include <math.h>

#define EPSILON 10e-6f

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

Network adamwOptimize(AdamState* state, Network net, Matrix gradient);
