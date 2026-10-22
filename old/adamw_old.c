#include "../matrix/matrix.h"
//#include "feedforward.h"
#include <math.h>
#include <stdlib.h>
#include "adamw_old.h"

Old_AdamwState old_adamwInit(Old_Network net, float learningRate, float weightDecay) {
	int layersAdjust = net.layers - 1;
	Old_AdamwState s = {
		(Old_AdamwParams) {learningRate, weightDecay, 0.9f, 0.999f, 0.0f},
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust),
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust)
	};  // Who needs error handling anyways?
	for (int i = 0; i < layersAdjust; i++) {
		old_stateAssign(&s.frstMomentWght[i], net.weights[i]); old_stateAssign(&s.scndMomentWght[i], net.weights[i]);
		old_stateAssign(&s.frstMomentBias[i], net.biases[i]); old_stateAssign(&s.scndMomentBias[i], net.biases[i]);
	} return s;
}

void old_stateAssign(Matrix* s, Matrix m) {*s = (Matrix) {m.w, m.h, m.n, (float*) calloc(m.n, sizeof(float))};}

void old_adamwFree(Old_AdamwState* state, Old_Network net) {
	for (int i = 0; i < net.layers - 1; i++) {
		free(state -> frstMomentWght[i].d); free(state -> scndMomentWght[i].d);
		free(state -> frstMomentBias[i].d); free(state -> scndMomentBias[i].d);
	}  // Free the matrices after freeing their data array
	free(state -> frstMomentWght); free(state -> scndMomentWght);
	free(state -> frstMomentBias); free(state -> scndMomentBias);
}

void old_adamwOptimize(Old_AdamwState* state, Old_Network* net, Matrix wghtGrad[], Matrix biasGrad[]) {
	state -> params.step++;
	float frstPowf = 1 - powf(state -> params.frstDecay, state -> params.step), scndPowf = 1 - powf(state -> params.scndDecay, state -> params.step);
	for (int i = 0; i < net->layers - 1; i++) {
		old_adamwStep(state -> params, state -> frstMomentWght[i], state -> scndMomentWght[i], wghtGrad[i], net->weights[i], frstPowf, scndPowf, 1);
		old_adamwStep(state -> params, state -> frstMomentBias[i], state -> scndMomentBias[i], biasGrad[i], net->biases[i], frstPowf, scndPowf, 0);
	}
}

void old_adamwStep(Old_AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix w1, float frstPowf, float scndPowf, int w) {
	float frstBias, scndBias;
	for (int i = 0; i < f.n; i++) {
		// m1 = b1 * m0 + (1 - b1) * g1
		f.d[i] = p.frstDecay * f.d[i] + (1.0f - p.frstDecay) * g.d[i];
		// v1 = b2 * v0 + (1 - b2) * g1^2
		s.d[i] = p.scndDecay * s.d[i] + (1.0f - p.scndDecay) * g.d[i] * g.d[i];
		// mhat1 = m1 / (1 - b1^t);
		frstBias = f.d[i] / frstPowf;
		// vhat1 = v1 / (1 - b2^t);
		scndBias = s.d[i] / scndPowf;
		// theta1 = theta0 - alpha X mhat1 / (sqrt(vhat1) + epsilon)
		float old = w1.d[i];
		w1.d[i] = old - p.learningRate * frstBias / (sqrtf(scndBias) + EPSILON);
		// theta1 = theta1  - alpha * lambda * theta0 (if NOT the bias)
		if (w) {w1.d[i] = w1.d[i] - p.learningRate * p.wghtDecay * old;}
	}
}
