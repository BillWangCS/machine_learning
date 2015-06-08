#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>

#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUMBER 1024
#define MAX_CLASS_NUMBER 8

struct KNN{
	double distance;
	int label;
};

bool knn_cmp(KNN a, KNN b){
	return a.distance < b.distance;
}

double compute_distance(double sample1[], double sample2[], int feature_number){
	double result = 0;
	for (int i = 0; i < feature_number; i++){
		result += (sample1[i] - sample2[i]) * (sample1[i] - sample2[i]);
	}
	return sqrt(result);
}

int k_nearest_neighbors_predict(double train[][MAX_FEATURE_DIMENSION],
								int label[],
								int sample_number,
								int K,
								double test[],
								int feature_number){
	std::vector<KNN> knn_vec;
	for (int i = 0; i < sample_number; i++){
		KNN temp;
		temp.distance = compute_distance(train[i], test, feature_number);
		temp.label = label[i];
		knn_vec.push_back(temp);
	}
	sort(knn_vec.begin(), knn_vec.end(), knn_cmp);
	int class_count[MAX_CLASS_NUMBER] = {0};
	for (int i = 0; i < K; i++){
		class_count[knn_vec[i].label]++;
	}
	int result = 0;
	int max_count  = -1;
	for (int i = 0; i < MAX_CLASS_NUMBER; i++){
		if (max_count < class_count[i]){
			result = i;
			max_count = class_count[i];
		}
	}
	return result;
}

void split_data_to_train_and_test(double X[][MAX_FEATURE_DIMENSION],
									int y[MAX_SAMPLE_NUMBER],
									int sample_number,
									int feature_number,
									double train[][MAX_FEATURE_DIMENSION],
									int label[],
									double test[],
									int sample_id){
	for (int i = 0; i < feature_number; i++){
		test[i] = X[sample_id][i];
	}
	for (int i = 0; i < sample_id; i++){
		for (int j = 0; j < feature_number; j++){
			train[i][j] = X[i][j];
		}
		label[i] = y[i];
	}
	for (int i = sample_id + 1; i < sample_number; i++){
		for (int j = 0; j < feature_number; j++){
			train[i-1][j] = X[i][j];
		}
		label[i-1] = y[i];
	}
}

double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int y[MAX_SAMPLE_NUMBER];
double train[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION];
int label[MAX_SAMPLE_NUMBER];
double test[MAX_FEATURE_DIMENSION];

int main(){
	int feature_number;
	int sample_number;
	scanf("%d %d", &feature_number, &sample_number);
	for (int i = 0; i < sample_number; i++){
		for (int j = 0; j < feature_number; j++){
			scanf("%lf", &X[i][j]);
		}
		scanf("%d", &y[i]);
	}

	int best_K = 0;
	int max_right_count = -1;

	for (int K = 1; K <= sample_number; K++){
		int right_count = 0;
		for (int i = 0; i < sample_number; i++){
			split_data_to_train_and_test(X, y, sample_number,
										feature_number, train, label, test, i);
			int result = k_nearest_neighbors_predict(train, label,
										sample_number-1, K, test, feature_number);
			if (result == y[i]){
				right_count++;
			}
		}
		if (max_right_count <= right_count){
			max_right_count = right_count;
			best_K = K;
		}
		printf("K = %d, there is %d right predict.\n", K, right_count);
	}
	printf("The best K is %d, there is %d right predict.\n", best_K, max_right_count);
    return 0;
}
