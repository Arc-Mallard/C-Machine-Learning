#ifndef ADAMW_H
#define ADAMW_H

#include "types_old.h"

void old_stateAssign(Matrix* s, Matrix m);

Old_AdamwState old_adamwInit(Old_Network net, float learningRate, float weightDecay);

void old_adamwFree(Old_AdamwState* state, Old_Network net);

void old_adamwStep(Old_AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix w1, float frstPowf, float scndPowf, int w);

void old_adamwOptimize(Old_AdamwState* state, Old_Network* net, Matrix wghtGrad[], Matrix biasGrad[]);

#endif
