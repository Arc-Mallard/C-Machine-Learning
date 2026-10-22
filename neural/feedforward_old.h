#ifndef FEEDFORWARD_OLD_H
#define FEEDFORWARD_OLD_H

#include "../types.h"

// ALL matrices below should be treated as vectors
Old_Network old_init(int layers, int* size, float deviation);

Old_Network old_initEmptyNet(int layers, int* size);

Matrix old_forward(Old_Network net, Matrix input);

void old_backward(AdamwState* adamw, Old_Network* net, Matrix output, Matrix expected);

void old_train(Old_Network* net, Matrix inputs[], Matrix expected[], int length, int epochs);

Matrix old_predict(Old_Network net, Matrix input);

void old_freeNet(Old_Network* net);

void old_printNet(Old_Network net);

unsigned char* old_serializeOld_Network(Old_Network n);

Old_Network old_deserializeOld_Network(unsigned char* m);

#endif
