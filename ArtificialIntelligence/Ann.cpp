#include <iostream>
#include <cmath>
#include <ctime>

#include "Ann.h"

using namespace std;

Ann::Ann(int L, int *S)
{
	_L = L;
	_S = new int[L];

	for (int i = 0; i < L; i++)
	{
		_S[i] = S[i];
	}

	_x = new double*[L];
	_y = new double*[L];

	for (int j = 0; j < L; j++)
	{
		_x[j] = new double[S[j]];
		_y[j] = new double[S[j]];

		for (int i = 0; i < S[j]; i++)
		{
			_x[j][i] = 0;
			_y[j][i] = 0;
		}
	}

	_w = new double**[L - 1];
	srand(time(NULL));

	for (int j = 0; j < L - 1; j++)
	{
		_w[j] = new double*[S[j]];

		for (int k = 0; k < S[j]; k++)
		{
			_w[j][k] = new double[S[j + 1]];

			for (int i = 0; i < S[j + 1]; i++)
			{
				_w[j][k][i] = 2 * rand() / (double)RAND_MAX - 1;	// Initialize the weights with values in [-1, 1].
			}
		}
	}
}

Ann::~Ann()
{
	for (int j = 0; j < _L; j++)
	{
		delete[] _x[j];
		delete[] _y[j];
	}

	delete[] _x;
	delete[] _y;

	for (int j = 0; j < _L - 1; j++)
	{
		for (int k = 0; k < _S[j]; k++)
		{
			delete[] _w[j][k];
		}

		delete[] _w[j];
	}

	delete[] _w;
	delete[] _S;
}

void Ann::Evaluate(double *input, double *output)
{
	for (int i = 0; i < _S[0]; i++)
	{
		_x[0][i] = input[i];	// This array is purely formal.
		_y[0][i] = _x[0][i];
	}

	for (int j = 1; j < _L; j++)
	{
		for (int i = 0; i < _S[j]; i++)
		{
			_x[j][i] = 0;

			for (int k = 0; k < _S[j - 1]; k++)
			{
				_x[j][i] += _w[j - 1][k][i] * _y[j - 1][k];
			}

			_y[j][i] = _Sigma(_x[j][i]);	// A bias is yet to be implemented.
		}
	}

	if (output != NULL)
	{
		for (int i = 0; i < _S[_L - 1]; i++)
		{
			output[i] = _y[_L - 1][i];
		}
	}
}

void Ann::Train(double *input, double *expectedOutput, double eta)
{
	Evaluate(input, NULL);

	double **yP = new double*[_L];

	for (int j = 0; j < _L; j++)
	{
		yP[j] = new double[_S[j]];
	}

	for (int i = 0; i < _S[_L - 1]; i++)
	{
		yP[_L - 1][i] = _y[_L - 1][i] - expectedOutput[i];
	}

	for (int j = _L - 2; j >= 0; j--)
	{
		for (int i = 0; i < _S[j]; i++)
		{
			yP[j][i] = 0;

			for (int k = 0; k < _S[j + 1]; k++)
			{
				yP[j][i] += yP[j + 1][k] * _SigmaPrime(_x[j + 1][k]) * _w[j][i][k];
			}
		}
	}

	for (int j = 0; j < _L - 1; j++)
	{
		for (int k = 0; k < _S[j]; k++)
		{
			for (int i = 0; i < _S[j + 1]; i++)
			{
				_w[j][k][i] -= eta * yP[j + 1][i] * _SigmaPrime(_x[j + 1][i]) * _y[j][k];
			}
		}
	}

	for (int j = 0; j < _L; j++)
	{
		delete[] yP[j];
	}

	delete[] yP;
}

// This is the sigmoid function.

double Ann::_Sigma(double x)
{
	return 1 / (1 + exp(-x));
}

double Ann::_SigmaPrime(double x)
{
	return exp(x) / pow(1 + exp(x), 2);
}