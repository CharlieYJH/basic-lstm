#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	LSTM network(128, 128, 0.001);
	network.load("./samples/shakespear.txt");
	network.loadState("./data/weights.txt");
	network.saveTo("./data/weights.txt");
	network.train(1, 100);
	network.output(10000);
	std::cout << std::endl;
	return 0;
}
