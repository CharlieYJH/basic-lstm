#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	LSTM network(60, 60, 0.001);
	network.load("./samples/samples.txt");
	network.train(1000, 10);
	network.output(400);
	return 0;
}
