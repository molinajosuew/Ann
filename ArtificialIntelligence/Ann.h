#pragma once

class Ann
{
public:
	Ann(int L, int *S);
	~Ann();
	void Evaluate(double *input, double *output);
	void Train(double *input, double *expectedOutput, double eta);

private:
	int _L;
	int *_S;
	double **_x;
	double **_y;
	double ***_w;

	double _Sigma(double x);
	double _SigmaPrime(double x);
};