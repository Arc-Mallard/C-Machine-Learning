#ifndef ADAMW_H
#define ADAMW_H

#include "types.h"

void stateAssign(Matrix* s, Matrix m);

AdamwState adamwInit(Network net, float learningRate, float weightDecay);

void adamwFree(AdamwState* state, Network net);

void adamwStep(AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix w1, float frstPowf, float scndPowf, int w);

void adamwOptimize(AdamwState* state, Network* net, Matrix wghtGrad[], Matrix biasGrad[]);

#endif
