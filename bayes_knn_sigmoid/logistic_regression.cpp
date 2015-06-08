#include <stdio.h>
#include <math.h>

#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUMBER 1024
#define MAX_ITERATE_NUMBER 1024

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

double compute_cost(double X[][MAX_FEATURE_DIMENSION], int y[], double theta[], int feature_number,int sample_number){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		double h = hypothesis(X[i], theta, feature_number);
		sum += -y[i] * log(h) - (1 - y[i]) * log(1 - h);
	}
	return sum/sample_number;
}

double compute_gradient(double X[][MAX_FEATURE_DIMENSION], int y[], double theta[], int feature_number, int feature_pos, int sample_number){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		double h = hypothesis(X[i], theta, feature_number);
		sum += (h - y[i]) * X[i][feature_pos];
	}
	return sum/sample_number;
}

void gradient_descent(double X[][MAX_FEATURE_DIMENSION], int y[],
	double theta[], int feature_number, int sample_number,
	double alpha, int iterate_number, double J[]){
	for (int i = 0; i < iterate_number; i++){
		double temp[MAX_FEATURE_DIMENSION] = {0};
		for (int j = 0; j <= feature_number; j++){
			temp[j] = theta[j] - alpha * compute_gradient(X, y, theta, feature_number, j, sample_number);
		}
		for (int j = 0; j <= feature_number; j++){
			theta[j] = temp[j];
		}
		J[i] = compute_cost(X, y, theta, feature_number, sample_number);
	}
}

double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int y[MAX_SAMPLE_NUMBER];
double J[MAX_ITERATE_NUMBER];
double theta[MAX_FEATURE_DIMENSION] = {0};

int main(){
	int feature_number;
	int sample_number;
	double alpha;
	int iterate_number;
	scanf("%d %d %lf %d", &feature_number, &sample_number, &alpha, &iterate_number);
	for (int i = 0; i < sample_number; i++){
		X[i][0] = 1;
		for (int j = 1; j <= feature_number; j++){
			scanf("%lf", &X[i][j]);
		}
		scanf("%d", &y[i]);
	}

	gradient_descent(X, y, theta, feature_number, sample_number, alpha, iterate_number, J);

	for (int i = 0; i < iterate_number; i++){
		printf("%lf\n", J[i]);
	}
	for (int i = 0; i < feature_number; i++){
		printf("%lf ", theta[i]);
	}
	printf("%lf\n", theta[feature_number]);
    return 0;
}
