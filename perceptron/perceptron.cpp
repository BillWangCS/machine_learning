#include <stdio.h>

#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUMBER 1024

int compute_activation(double x[], double weights[], int feature_number){
    double sum = 0;
    for (int i = 0; i <= feature_number; i++){
        sum += x[i]*weights[i];
    }
    if (sum > 0){
        return 1;
    }
    return 0;
}

void perceptron_train(double X[][MAX_FEATURE_DIMENSION], int y[]
    , double weights[], int feature_number, int sample_number, double alpha, int iterate_number){
    for (int i = 0; i < sample_number; i++){
        X[i][0] = 1;
    }
    for (int i = 0; i < iterate_number; i++){
        double delta[MAX_FEATURE_DIMENSION] = {0};
        for (int j = 0; j < sample_number; j++){
            int activation = compute_activation(X[j], weights, feature_number);
            for (int k = 0; k <= feature_number; k++){
                delta[k] += alpha * (y[j] - activation) * X[j][k];
            }
        }
        for (int j = 0; j <= feature_number; j++){
            weights[j] += delta[j];
        }
        for (int j = 0; j < feature_number; j++){
            printf("%.3lf ", weights[j]);
        }
        printf("%.3lf\n", weights[feature_number]);
    }
}

double weights[MAX_FEATURE_DIMENSION] = {0};
double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION] = {0};
int y[MAX_SAMPLE_NUMBER] = {0};

int main(){    
    int feature_number = 0;
    int sample_number = 0;
    int iterate_number = 0;
    double alpha = 0;
    scanf("%d %d %lf %d", &feature_number, &sample_number, &alpha, &iterate_number);
    for (int i = 0; i < sample_number; i++){
        for (int j = 1; j <= feature_number; j++){
            scanf("%lf", &X[i][j]);
        }
        scanf("%d", &y[i]);
    }
    for (int i = 0; i <= feature_number; i++){
        scanf("%lf", &weights[i]);
    }
    perceptron_train(X, y, weights, feature_number, sample_number, alpha, iterate_number);
    return 0;
}
