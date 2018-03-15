#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	LSTM network(60, 60, 0.05);
	network.load("./samples/samples.txt");
	network.train(1000, 50);
	return 0;
}
