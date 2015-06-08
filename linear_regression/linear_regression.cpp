#include <stdio.h>
#include <math.h>

#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUMBER 1024
#define MAX_ITERATE_NUMBER 1024

double standard_deviation(double X[], int sample_number, double average){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		sum += (X[i] - average) * (X[i] - average);
	}
	sum = sum / (sample_number-1);
	return sqrt(sum);
}

void feature_normalize(double X[][MAX_FEATURE_DIMENSION], int feature_number,int sample_number){
	double average[MAX_FEATURE_DIMENSION] = {0};
	double range[MAX_FEATURE_DIMENSION] = {0};
	for (int i = 1; i <= feature_number; i++){
		double sum = 0;
		double temp_X[MAX_SAMPLE_NUMBER] = {0};
		for (int j = 0; j < sample_number; j++){
			sum += X[j][i];
			temp_X[j] = X[j][i];
		}
		average[i] = sum / sample_number;
		range[i] = standard_deviation(temp_X, sample_number, average[i]);
	}
	for (int i = 0; i < sample_number; i++){
		for (int j = 1; j <= feature_number; j++){
			X[i][j] = (X[i][j] - average[j]) / range[j];
		}
	}
	for (int i = 0; i < sample_number; i++){
		X[i][0] = 1;
	}
}

double compute_cost(double X[][MAX_FEATURE_DIMENSION], int y[], double theta[], int feature_number,int sample_number){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		double res = 0;
		for (int j = 0; j <= feature_number; j++){
			res +=  X[i][j] * theta[j];
		}
		sum += (res - y[i]) * (res - y[i]);
	}
	return sum/(2*sample_number);
}

double compute_gradient(double X[][MAX_FEATURE_DIMENSION], int y[], double theta[], int feature_number, int feature_pos, int sample_number){
	double sum = 0;
	for (int i = 0; i < sample_number; i++){
		double res = 0;
		for (int j = 0; j <= feature_number; j++){
			res +=  X[i][j]*theta[j];
		}
		sum += (res - y[i]) * X[i][feature_pos];
	}
	return sum/sample_number;
}

void gradient_descent(double X[][MAX_FEATURE_DIMENSION], int y[]
	, double theta[], int feature_number, int sample_number, double alpha, int iterate_number, double J[]){
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
		for (int j = 1; j <= feature_number; j++){
			scanf("%lf", &X[i][j]);
		}
		scanf("%d", &y[i]);
	}
	feature_normalize(X, feature_number, sample_number);
	gradient_descent(X, y, theta, feature_number, sample_number, alpha, iterate_number, J);

	for (int i = 0; i < iterate_number; i++){
		printf("%.3lf\n", J[i]);
	}
	for (int i = 0; i <= feature_number; i++){
		printf("%.3lf ", theta[i]);
	}
	printf("\n");
    return 0;
}
