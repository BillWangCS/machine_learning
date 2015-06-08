
#include <stdio.h>
#include <math.h>

double sigmoid(double z){
	return 1 / (1 + exp(-z));
}

double hypothesis(double x[], double theta[], int feature_number){
	double h = 0;
	for (int i = 0; i <= feature_number; i++){
		h += x[i] * theta[i];
	}
	return sigmoid(h);
}

#define MAX_FEATURE_DIMENSION 128
#define MAX_LABEL_NUMBER 12

void forward_propagation(double input[],
						 int feature_number,
						 double W[][MAX_FEATURE_DIMENSION],
						 int neuron_num,
						 double a[]){

	for (int i = 0; i < neuron_num; i++){
		a[i+1] = hypothesis(input, W[i], feature_number);
	}
}

double compute_cost(double X[][MAX_FEATURE_DIMENSION], 
					int y[],
					int feature_number,
					int sample_number,
					double W1[][MAX_FEATURE_DIMENSION],
					int hidden_layer_size,
					double W2[][MAX_FEATURE_DIMENSION],
					int label_num,
					double a2[][MAX_FEATURE_DIMENSION],
					double a3[][MAX_FEATURE_DIMENSION]){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		X[i][0] = 1;
		forward_propagation(X[i], feature_number, W1, hidden_layer_size, a2[i]);
		a2[i][0] = 1;
		forward_propagation(a2[i], hidden_layer_size, W2, label_num, a3[i]);
		double yy[MAX_LABEL_NUMBER] = {0};
		yy[y[i]] = 1;
		for (int j = 1; j <= label_num; j++){
			sum += -yy[j] * log(a3[i][j]) - (1 - yy[j]) * log(1 - a3[i][j]);
		}
	}
	return sum / sample_number;
}

int main(){
	double X[][MAX_FEATURE_DIMENSION] = {
		{0, 0.084147, 0.090930},
		{0, 0.090930, 0.065699},
		{0, 2, 3}
	};
	int y[] = {1, 2, 2};
	int hidden_layer_size = 4;
	int label_num = 2;
	int feature_number = 2;
	int sample_number = 3;
	double W1[][MAX_FEATURE_DIMENSION] = {
		{0.084147, -0.027942, -0.099999},
		{0.090930, 0.065699, -0.053657},
		{0.014112, 0.098936, 0.042017},
		{-0.075680, 0.041212, 0.099061},
	};
	double W2[][MAX_FEATURE_DIMENSION] = {
		{0.084147, -0.075680, 0.065699, -0.054402, 0.042017},
		{0.090930, -0.095892, 0.098936, -0.099999, 0.099061}
	};
	double a2[3][MAX_FEATURE_DIMENSION] = {0};
	double a3[3][MAX_FEATURE_DIMENSION] = {0};

	double J = compute_cost(X, y, feature_number, sample_number,
		W1, hidden_layer_size, W2, label_num, a2, a3);
		
	printf("J = %lf\n", J);
	printf("a2:\n");
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j <= hidden_layer_size; j++){
			printf("%lf ", a2[i][j]);
		}
		printf("\n");
	}
	printf("a3:\n");
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j <= label_num; j++){
			printf("%lf ", a3[i][j]);
		}
		printf("\n");
	}
	return 0;
}
