#include "adamw.h"

AdamwState adamwInit(Network net, float learningRate, float weightDecay) {
	int layersAdjust = net.layers - 1;
	AdamwState s = {
		(AdamwParams) {learningRate, weightDecay, 0.9f, 0.999f, 0.0f},
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust),
		malloc(sizeof(Matrix) * layersAdjust), malloc(sizeof(Matrix) * layersAdjust)
	};  // Who needs error handling anyways?
	for (int i = 0; i < layersAdjust; i++) {
		stateAssign(&s.frstMomentWght[i], net.weights[i]); stateAssign(&s.scndMomentWght[i], net.weights[i]);
		stateAssign(&s.frstMomentBias[i], net.biases[i]); stateAssign(&s.scndMomentBias[i], net.biases[i]);
	} return s;
}

void stateAssign(Matrix* s, Matrix m) {*s = (Matrix) {m.w, m.h, m.n, (float*) calloc(m.n, sizeof(float))};}

void adamwFree(AdamwState* state, Network net) {
	for (int i = 0; i < net.layers - 1; i++) {
		free(state -> frstMomentWght[i].d); free(state -> scndMomentWght[i].d);
		free(state -> frstMomentBias[i].d); free(state -> scndMomentBias[i].d);
	}  // Free the matrices after freeing their data array
	free(state -> frstMomentWght); free(state -> scndMomentWght);
	free(state -> frstMomentBias); free(state -> scndMomentBias);
}

Network adamwOptimize(AdamwState* state, Network net, Matrix wghtGrad[], Matrix biasGrad[]) {
	Network n = initEmptyNet(net.layers, net.size); state -> params.step++;
	freeMatrixArr(n.activations, net.layers); freeMatrixArr(n.preactivations, net.layers - 1);
	n.activations = net.activations; n.preactivations = net.preactivations;
	float frstPowf = 1 - powf(state -> params.frstDecay, state -> params.step), scndPowf = 1 - powf(state -> params.scndDecay, state -> params.step);
	for (int i = 0; i < net.layers - 1; i++) {
		adamwStep(state -> params, state -> frstMomentWght[i], state -> scndMomentWght[i], wghtGrad[i], &n.weights[i], net.weights[i], frstPowf, scndPowf, 1);
		adamwStep(state -> params, state -> frstMomentBias[i], state -> scndMomentBias[i], biasGrad[i], &n.biases[i], net.biases[i], frstPowf, scndPowf, 0);
	} return n;
}

void adamwStep(AdamwParams p, Matrix f, Matrix s, Matrix g, Matrix* w1, Matrix w2, float frstPowf, float scndPowf, int w) {
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
		// theta1 = theta0 - alpha X mhat1 / (sqrt(vhat1) + epsilon) - alpha * lambda *  theta0
		w1 -> d[i] = w2.d[i] - p.learningRate * frstBias / (sqrtf(scndBias) + EPSILON);
		if (w) {w1 -> d[i] = w1 -> d[i] - p.learningRate * p.wghtDecay * w2.d[i];}
	}
}
