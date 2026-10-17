// heavily "inspired" by https://github.com/loretoparisi/adamw.c/blob/main/adamw.c

#include "optimizer.h"

AdamState adamInit(Network net, float learningRate, float weightDecay) {
	AdamState s = {
		(AdamParams) {learningRate, weightDecay, 0.9, 0.999, 0},
		malloc(sizeof(Matrix) * net.layers), malloc(sizeof(Matrix) * net.layers),
		malloc(sizeof(Matrix) * net.layers), malloc(sizeof(Matrix) * net.layers)
	};  // Who needs error handling anyways?
	for (int i = 0; i < net.layers; i++) {
		stateAssign(&s.frstMomentWght[i], net.weights[i]); stateAssign(&s.scndMomentWght[i], net.weights[i]);
		stateAssign(&s.frstMomentBias[i], net.biases[i]); stateAssign(&s.scndMomentBias[i], net.biases[i]);
	} return s;
}

void adamFree(AdamState* state, Network net) {
	for (int i = 0; i < net.layers; i++) {
		free(state -> frstMomentWght); free(state -> scndMomentWght);
		free(state -> frstMomentBias); free(state -> scndMomentBias);
	}  // Free the matrices after freeing their data array
	free(state -> frstMomentWght); free(state -> scndMomentWght);
	free(state -> frstMomentBias); free(state -> scndMomentBias);
}

Network adamwOptimize(AdamState* state, Network net, Matrix gradient[]) {
	Network n = initEmptyNet(net.layers, net.size); state -> params.step++;
	n.activations = net.activations; n.preactivations = net.preactivations;

	float frstBias, scndBias;
	for (int i = 0; i < net.layers - 1; i++) {
		for (int j = 0; j < state -> frstMomentWght[i].n; j++) {
			// m1 = b1 * m0 + (1 - b1) * g1
			state -> frstMomentWght[i].d[j] = state -> params.frstDecay * state -> frstMomentWght[i].d[j] + (1 - state -> params.frstDecay) * gradient[i].d[j];
			// v1 = b2 * v0 + (1 - b2) * g1^2
			state -> scndMomentWght[i].d[j] = state -> params.scndDecay * state -> scndMomentWght[i].d[j] + (1 - state -> params.scndDecay) * gradient[i].d[j] * gradient[i].d[j];
			// mhat1 = m1 / (1 - b1^t);
			frstBias = state -> frstMomentWght[i].d[j] / (1 - powf(state -> params.frstDecay, state -> params.step));
			// vhat1 = v1 / (1 - b2^t);
			scndBias = state -> scndMomentWght[i].d[j] / (1 - powf(state -> params.scndDecay, state -> params.step));
			// theta1 = theta0 - alpha X mhat1 / (sqrt(vhat1) + epsilon) - alpha * lambda *  theta0
			n.weights[i].d[j] = net.weights[i].d[j] - state -> params.learningRate * frstBias / (sqrtf(scndBias) + EPSILON) - state -> params.learningRate * state -> params.wghtDecay * net.weights[i].d[j];
		}
	}

	return n;
}

void stateAssign(Matrix* s, Matrix m) {
	*s = (Matrix) {m.w, m.h, m.n, (float*) calloc(m.n, sizeof(float))};
}
