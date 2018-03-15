#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	LSTM network(60, 60, 0.05);
	network.load("./samples/samples.txt");
	// network.train(500, 50);
	// network.output(100);
	return 0;
}
