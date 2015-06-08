
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
	return h;
}

void forward_propagation(double input[],
						 int feature_number,
						 double W[][MAX_FEATURE_DIMENSION],
						 int neuron_num,
						 double z[],
						 double a[]){

	for (int i = 0; i < neuron_num; i++){
		z[i+1] = hypothesis(input, W[i], feature_number);
		a[i+1] = sigmoid(z[i+1]);
	}
}

double sigmoid_gradient(double z){
	return sigmoid(z) * (1 - sigmoid(z));
}

void compute_layer_error(double layer_error[],
						double W[][MAX_FEATURE_DIMENSION],
						int neuron_num,
						int feature_number,
						double next_layer_error[],
						double z[]){

	for (int i = 1; i <= feature_number; i++){
		for (int j = 0; j < neuron_num; j++){
			layer_error[i] += W[j][i] * next_layer_error[j + 1];
		}
	}
	for (int i = 1; i <=feature_number; i++){
		layer_error[i] = layer_error[i] * sigmoid_gradient(z[i]);
	}
}

void accumulate_gradient(double sum[][MAX_FEATURE_DIMENSION], 
						 double layer_error[],
						 int neuron_num,
						 int feature_number,
						 double a[]){
	for (int i = 0; i < neuron_num; i++){
		for (int j = 0; j <= feature_number; j++){
			sum[i][j] += layer_error[i+1] * a[j];
		}
	}
}

void compute_gradient(double X[][MAX_FEATURE_DIMENSION], 
						int y[],
						int feature_number,
						int sample_number,
						double W1[][MAX_FEATURE_DIMENSION],
						int hidden_layer_size,
						double W2[][MAX_FEATURE_DIMENSION],
						int label_num,
						double w1_grad[][MAX_FEATURE_DIMENSION],
						double w2_grad[][MAX_FEATURE_DIMENSION]){

	double grad1_sum[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION] = {0};
	double grad2_sum[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION] = {0};
	for (int i = 0; i < sample_number; i++){
		X[i][0] = 1;
		double z2[MAX_FEATURE_DIMENSION] = {0, 0};
		double a2[MAX_FEATURE_DIMENSION] = {1, 0};		
		forward_propagation(X[i], feature_number, W1, hidden_layer_size, z2, a2);
		double z3[MAX_FEATURE_DIMENSION] = {0};
		double a3[MAX_FEATURE_DIMENSION] = {0};		
		forward_propagation(a2, hidden_layer_size, W2, label_num, z3, a3);
		double yy[MAX_LABEL_NUMBER] = {0};
		yy[y[i]] = 1;
		double layer3_error[MAX_FEATURE_DIMENSION] = {0};
		for (int j = 1; j <= label_num; j++){
			layer3_error[j] = a3[j] - yy[j];
		}
		double layer2_error[MAX_FEATURE_DIMENSION] = {0};
		compute_layer_error(layer2_error, W2, label_num, hidden_layer_size, layer3_error, z2);
		accumulate_gradient(grad2_sum, layer3_error, label_num, hidden_layer_size, a2);
		accumulate_gradient(grad1_sum, layer2_error, hidden_layer_size, feature_number, X[i]);
	}

	for (int i = 0; i < hidden_layer_size; i++){
		for (int j = 0; j <= feature_number; j++){
			w1_grad[i][j] = grad1_sum[i][j] / sample_number;
		}
	}

	for (int i = 0; i < label_num; i++){
		for (int j = 0; j <= hidden_layer_size; j++){
			w2_grad[i][j] = grad2_sum[i][j] / sample_number;
		}
	}
}

double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int y[MAX_SAMPLE_NUMBER];
double W1[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION];
double W2[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION];
double a2[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
double a3[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
double w1_grad[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION] = {0};
double w2_grad[MAX_FEATURE_DIMENSION][MAX_FEATURE_DIMENSION] = {0};

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
	compute_gradient(X, y, feature_number, 3, W1,
					hidden_layer_size, W2, label_num, w1_grad, w2_grad);
	for (int i = 0; i < hidden_layer_size; i++){
		for (int j = 0; j < feature_number; j++){
			printf("%lf ", w1_grad[i][j]);
		}
		printf("%lf\n", w1_grad[i][feature_number]);
	}
	for (int i = 0; i < label_num; i++){
		for (int j = 0; j < hidden_layer_size; j++){
			printf("%lf ", w2_grad[i][j]);
		}
		printf("%lf\n", w2_grad[i][hidden_layer_size]);
	}
	return 0;
}
