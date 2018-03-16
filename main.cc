#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	LSTM network(128, 128, 0.0005);
	network.load("./samples/samples.txt");
	network.loadState("./data/weights.txt");
	network.train(500, 100);
	network.output(100);
	network.saveState("./data/weights.txt");
	std::cout << std::endl;
	return 0;
}
