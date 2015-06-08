
#include <stdio.h>
#include <math.h>

#define MAX_SAMPLE_NUMBER 1024
#define MAX_FEATURE_DIMENSION 128
#define MAX_LABEL_NUMBER 12

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

void forward_propagation(double a[],
						 int feature_number,
						 double W[][MAX_FEATURE_DIMENSION],
						 int neuron_num,
						 double output[]){

	for (int i = 0; i < neuron_num; i++){
		output[i+1] = hypothesis(a, W[i], feature_number);
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

double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int y[MAX_SAMPLE_NUMBER];
double W1[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION];
double W2[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION];
double a2[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
double a3[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];

int main(){
	int feature_number;
	int sample_number;
	int hidden_layer_size;
	int label_num;
	scanf("%d %d %d %d", &feature_number, &sample_number, &hidden_layer_size, &label_num);
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j <= feature_number; j++){
			scanf("%lf", &X[i][j]);
		}
		scanf("%d", &y[i]);
	}
	for (int i = 0; i < hidden_layer_size; i++){
		for (int j = 0; j <= feature_number; j++){
			scanf("%lf", &W1[i][j]);
		}
	}
	for (int i = 0; i < label_num; i++){
		for (int j = 0; j <= hidden_layer_size; j++){
			scanf("%lf", &W2[i][j]);
		}
	}
	double J = compute_cost(X, y, feature_number, sample_number,
		W1, hidden_layer_size, W2, label_num, a2, a3);
	printf("%lf\n", J);
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j < hidden_layer_size; j++){
			printf("%lf ", a2[i][j]);
		}
		printf("%lf\n", a2[i][hidden_layer_size]);
	}
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j < label_num; j++){
			printf("%lf ", a3[i][j]);
		}
		printf("%lf\n", a3[i][label_num]);
	}
	return 0;
}
