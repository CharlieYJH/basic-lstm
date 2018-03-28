#define NDEBUG
#define EIGEN_NO_DEBUG
#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	std::srand(std::time(0));
	LSTM network(256, 0.01);
	network.load("./samples/shakespear.txt");
	network.loadState("./data/weights.txt");
	network.setSoftmaxTemperature(4.0);
	// network.saveTo("./data/weights.txt");
	// network.train(300, 100, 100, 25);
	// network.output(1000);
	network.beamSearchOutput(5, 5000);
	return 0;
}
