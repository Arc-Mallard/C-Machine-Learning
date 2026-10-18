#include "../matrix/matrix.h"
#include "activations.h"

// ALL matrices below should be treated as vectors
Network init(int size[]);

Matrix forward(Network net, Matrix input);

Network backward(Network net, Matrix output, Matrix expected);

Network train(Network net, Matrix inputs[], Matrix expected[], int epochs);

Matrix predict(Network net, Matrix input);

unsigned char* serializeNetwork(Network n);

Network deserializeNetwork(unsigned char* m);
