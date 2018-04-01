#include <iostream>
#include <ctime>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	std::srand(std::time(0));
	LSTM network(256, 0.001);
	network.load("./samples/shakespear.txt");
	network.loadState("./data/weights.txt");
	network.setSoftmaxTemperature(3.0);
	// network.saveStateTo("./data/weights.txt");
	// network.train(300, 100, 100, 25);
	std::cout << network.beamSearchOutput(4, 5000) << std::endl;
	return 0;
}
