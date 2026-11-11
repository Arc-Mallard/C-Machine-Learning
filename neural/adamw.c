#include "adamw.h"
#include "../matrix/matrix.h"
//#include "feedforward.h"
#include <math.h>
#include <stdlib.h>

AdamwState adamwInit(Network net, float learningRate, float weightDecay) {
	int layersAdjust = net.layers - 1;
	AdamwState s = {
		(AdamwParams) {learningRate, weightDecay, 0.9f, 0.999f, 0.0f},
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust),
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust),
		// Swiglu Additions
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust),
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust)
	};  // Who needs error handling anyways?
	for (int i = 0; i < layersAdjust; i++) {
		stateAssign(&s.frstMomentWght[i], net.weights1[i]); stateAssign(&s.scndMomentWght[i], net.weights1[i]);
		stateAssign(&s.frstMomentBias[i], net.biases1[i]); stateAssign(&s.scndMomentBias[i], net.biases1[i]);
		// Swiglu additions
		stateAssign(&s.frstMomentWght2[i], net.weights2[i]); stateAssign(&s.scndMomentWght2[i], net.weights2[i]);
		stateAssign(&s.frstMomentBias2[i], net.biases2[i]); stateAssign(&s.scndMomentBias2[i], net.biases2[i]);
	} return s;
}

void stateAssign(Matrix* s, Matrix m) {*s = (Matrix) {m.w, m.h, m.n, (float*) calloc(m.n, sizeof(float))};}

void adamwFree(AdamwState* state, Network net) {
	for (int i = 0; i < net.layers - 1; i++) {
		free(state -> frstMomentWght[i].d); free(state -> scndMomentWght[i].d);
		free(state -> frstMomentBias[i].d); free(state -> scndMomentBias[i].d);
		// Swiglu additions
		free(state -> frstMomentWght2[i].d); free(state -> scndMomentWght2[i].d);
		free(state -> frstMomentBias2[i].d); free(state -> scndMomentBias2[i].d);
	}  // Free the matrices after freeing their data array
	free(state -> frstMomentWght); free(state -> scndMomentWght);
	free(state -> frstMomentBias); free(state -> scndMomentBias);
	// Swiglu additions
	free(state -> frstMomentWght2); free(state -> scndMomentWght2);
	free(state -> frstMomentBias2); free(state -> scndMomentBias2);
}

void adamwOptimize(AdamwState* state, Network* net, Matrix wghtGrad[], Matrix biasGrad[], Matrix wghtGrad2[], Matrix biasGrad2[]) {
	state -> params.step++;
	float frstPowf = 1 - powf(state -> params.frstDecay, state -> params.step), scndPowf = 1 - powf(state -> params.scndDecay, state -> params.step);
	for (int i = 0; i < net->layers - 1; i++) {
		adamwStep(state -> params, state -> frstMomentWght[i], state -> scndMomentWght[i], wghtGrad[i], net -> weights1[i], frstPowf, scndPowf, 1);
		adamwStep(state -> params, state -> frstMomentBias[i], state -> scndMomentBias[i], biasGrad[i], net -> biases1[i], frstPowf, scndPowf, 0);
		// Swiglu additions
		adamwStep(state -> params, state -> frstMomentWght2[i], state -> scndMomentWght2[i], wghtGrad2[i], net -> weights2[i], frstPowf, scndPowf, 1);
		adamwStep(state -> params, state -> frstMomentBias2[i], state -> scndMomentBias2[i], biasGrad2[i], net -> biases2[i], frstPowf, scndPowf, 0);
	}
}

void adamwStep(AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix w1, float frstPowf, float scndPowf, int w) {
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
