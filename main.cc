#define EIGEN_NO_DEBUG
#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	std::srand(std::time(0));
	LSTM network(128, 128, 128, 0.002);
	network.load("./samples/shakespear.txt");
	network.loadState("./data/weights.txt");
	network.saveTo("./data/weights.txt");
	network.train(50, 100);
	network.output(300);
	return 0;
}
