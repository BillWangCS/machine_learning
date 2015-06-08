#include <stdio.h>

#define MAX_CLASS_NUMBER 8
#define MAX_FEATURE_VALUE_NUMBER 32
#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUMBER 1024

void naive_bayesian_train(int X[][MAX_FEATURE_DIMENSION],
                          int y[],
                          int feature_number,
                          int sample_number,
                          double p_feature_value[][MAX_FEATURE_VALUE_NUMBER][MAX_CLASS_NUMBER],
                          double p_class[]){
    int class_count[MAX_CLASS_NUMBER] = {0};
    for (int i = 0; i < sample_number; i++){
        class_count[y[i]]++;
    }
    for (int i = 0; i < MAX_CLASS_NUMBER; i++){
        p_class[i] = (double)class_count[i] / sample_number;
    }

    for (int i = 0; i < feature_number; i++){
        int feature_value_class_count[MAX_FEATURE_VALUE_NUMBER][MAX_CLASS_NUMBER] = {0};
        for (int j = 0; j < sample_number; j++){
            feature_value_class_count[X[j][i]][y[j]]++;
        }
        for (int j = 0; j < 3; j++){
            for (int k = 0; k < MAX_CLASS_NUMBER; k++){
                if (class_count[k]){
                    p_feature_value[i][j][k] = (double)feature_value_class_count[j][k] / class_count[k];
                }
            }
        }
    }
}

int naive_bayesian_predict(int x[],
                           int feature_number, 
                           double p[][MAX_FEATURE_VALUE_NUMBER][MAX_CLASS_NUMBER],
                           double p_class[],
                           double p_result[],
                           int class_number){
    int final_result = -1;
    double final_p = -1;
    for (int i = 0; i < class_number; i++){
        p_result[i] = p_class[i];
    }
    for (int i = 0; i < class_number; i++){
        for (int j = 0; j < feature_number; j++){
            p_result[i] = p_result[i] * p[j][x[j]][i];
        }
        if (final_p < p_result[i]){
            final_p = p_result[i];
            final_result = i;
        }
    }
    return final_result;
}

double p_feature_value[MAX_FEATURE_DIMENSION][MAX_FEATURE_VALUE_NUMBER][MAX_CLASS_NUMBER] = {0};
double p_class[MAX_CLASS_NUMBER] = {0};
int X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int y[MAX_SAMPLE_NUMBER];
int X_test[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];

int main(){
    int feature_number;
    int sample_number;
    int test_number;
    int class_number;
    scanf("%d %d %d %d", &feature_number, &sample_number, &test_number, &class_number);
    for (int i = 0; i < sample_number; i++){
        for (int j = 0; j < feature_number; j++){
            scanf("%d", &X[i][j]);
        }
        scanf("%d", &y[i]);
    }
    for (int i = 0; i < test_number; i++){
        for (int j = 0; j < feature_number; j++){
            scanf("%d", &X_test[i][j]);
        }
    }
    naive_bayesian_train(X, y, feature_number, sample_number, p_feature_value, p_class);
    for (int i = 0; i < test_number; i++){
        double p_result[MAX_CLASS_NUMBER] = {0};
        int final_result = naive_bayesian_predict(X_test[i], feature_number, p_feature_value,
                                                p_class, p_result, class_number);
        for (int j = 0; j < class_number; j++){
            printf("%lf ", p_result[j]);
        }
        printf("%d\n", final_result);
    }
    return 0;
}
