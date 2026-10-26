#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "types.h"

// ALL matrices below should be treated as vectors
Network init(int layers, int* size, float deviation);

Network initEmptyNet(int layers, int* size);

Matrix forward(Network net, Matrix input);

void backward(AdamwState* adamw, Network* net, Matrix output, Matrix expected);

void train(Network* net, Matrix inputs[], Matrix expected[], int length, int epochs);

Matrix predict(Network net, Matrix input);

void freeNet(Network* net);

void printNet(Network net);

unsigned char* serializeNet(Network n);

Network deserializeNet(unsigned char* m);

#endif
