#include <stdio.h>
#include <math.h>
#include <vector>
#include <set>

#define MAX_CLASS_NUMBER 8
#define MAX_FEATURE_VALUE_NUMBER 32

double compute_entropy(double p){
    if (p != 0){
        return -p * (log(p) / log(2));
    }
    return 0;
}

double calculate_after_spliting_entropy(const std::vector<int> &feature_value, 
                                         const std::vector<int> &class_value){
    int feature_value_count[MAX_FEATURE_VALUE_NUMBER] = {0};
    int feature_value_class_count[MAX_FEATURE_VALUE_NUMBER][MAX_CLASS_NUMBER] = {0};
    int sample_number = feature_value.size();	
    for (int i = 0; i < sample_number; i++){
        feature_value_count[feature_value[i]]++;
        feature_value_class_count[feature_value[i]][class_value[i]]++;
    }	
    double result = 0;
    for (int i = 0; i < MAX_FEATURE_VALUE_NUMBER; i++){
        if (feature_value_count[i] == 0){
            continue;
        }
        double entropy = 0;
        for (int j = 0; j < MAX_CLASS_NUMBER; j++){
            double p = (double)feature_value_class_count[i][j] / feature_value_count[i];
            entropy += compute_entropy(p);
        }
        result += (double)feature_value_count[i] / sample_number * entropy;
    }	
    return result;
}

void get_data_by_feature_value(const std::vector<std::vector<int> > &data,
                               const std::vector<int> &class_value,
                               const std::vector<int> &feature_name,
                               int feature_pos,
                               int feature_value,
                               std::vector<std::vector<int> > &new_data,
                               std::vector<int> &new_class_value,
                               std::vector<int> &new_feature_name){
    for (unsigned int i = 0; i < feature_pos; i++){
        new_feature_name.push_back(feature_name[i]);
    }
    for (unsigned int i = feature_pos + 1; i < feature_name.size(); i++){
        new_feature_name.push_back(feature_name[i]);
    }
    for (unsigned int i = 0; i < data.size(); i++){
        if (data[i][feature_pos] == feature_value){
            std::vector<int> temp_vec;
            for (unsigned int j = 0; j < feature_pos; j++){
                temp_vec.push_back(data[i][j]);
            }
            for (unsigned int j = feature_pos + 1; j < data[i].size(); j++){
                temp_vec.push_back(data[i][j]);
            }
            new_data.push_back(temp_vec);
            new_class_value.push_back(class_value[i]);
        }
    }
}

void find_best_feature_name(const std::vector<std::vector<int> > &data,
							const std::vector<int> &class_value, 
							const std::vector<int> &feature_name,
							int &best_feature_name,
							int &best_feature_name_pos){							
    std::vector<double> spliting_entropy;
    for (unsigned int j = 0; j < feature_name.size(); j++){
        std::vector<int> feature_value;
        for (unsigned int i = 0; i < data.size(); i++){
            feature_value.push_back(data[i][j]);
        }
		double ret = calculate_after_spliting_entropy(feature_value, class_value);
        spliting_entropy.push_back(ret);
    }	
    best_feature_name = feature_name[0];
    best_feature_name_pos = 0;
    double min_entropy = spliting_entropy[0];	
    for (unsigned int i = 1; i < spliting_entropy.size(); i++){
        if (min_entropy > spliting_entropy[i]){
            min_entropy = spliting_entropy[i];
            best_feature_name = feature_name[i];
            best_feature_name_pos = i;
        }
    }
}

int check_is_leaf(const std::vector<int> &class_value, int feature_number){
    int class_value_count[MAX_CLASS_NUMBER] = {0};
    int class_number = 0;
    for (unsigned int i = 0; i < class_value.size(); i++){
        if (class_value_count[class_value[i]] == 0){
            class_number++;
        }
        class_value_count[class_value[i]]++;
    }	
    int max_count = -1;
    int leaf_class = -1;
    for (int i = 0; i < MAX_CLASS_NUMBER; i++){
        if (max_count < class_value_count[i]){
            max_count = class_value_count[i];
            leaf_class = i;
        }
    }	
    if (class_number == 1 || feature_number == 0){
        return leaf_class;
    }	
    return -1;
}

struct Node{
    int feature_name;
    int type;
    Node *child[MAX_FEATURE_VALUE_NUMBER];
    Node(){
        feature_name = -1;
        type = -1;
        for (int i = 0; i < MAX_FEATURE_VALUE_NUMBER; i++){
            child[i] = 0;
        }
    }
};

Node* construct_desicion_tree(const std::vector<std::vector<int> > &data,
                            const std::vector<int> &class_value, 
                            const std::vector<int> &feature_name, 
                            std::vector<Node *> &tree){
    Node *node = new Node();
    tree.push_back(node);
    node->type = check_is_leaf(class_value, feature_name.size());
    if (node->type != -1){
        return node;
    }
    int best_feature_name;
    int best_feature_name_pos;
    find_best_feature_name(data, class_value, feature_name, best_feature_name, best_feature_name_pos);
    node->feature_name = best_feature_name;
    std::set<int> value_set;
    for (int i = 0; i < data.size(); i++){
        value_set.insert(data[i][best_feature_name_pos]);
    }
    std::set<int>::iterator it;
    for (it = value_set.begin(); it!=value_set.end(); it++){
        std::vector<std::vector<int> > new_data;
        std::vector<int> new_class_value;
        std::vector<int> new_feature_name;
        get_data_by_feature_value(data, class_value, feature_name, best_feature_name_pos, *it, new_data, new_class_value, new_feature_name);        
        node->child[*it] = construct_desicion_tree(new_data, new_class_value, new_feature_name, tree);
    }
    return node;
}

void preorder_decision_tree(Node *node, int value, int level){
    for (int i = 0; i < level; i++){
        printf("---");
    }
    if (level){
        printf("%d->", value);
    }    
    if (node->feature_name == -1) {
        printf("leaf(%d)\n", node->type);
    }
    else{
        printf("inner(%d)\n", node->feature_name);
    }
    for (int i = 0; i < MAX_FEATURE_VALUE_NUMBER; i++){
        if (node->child[i]){
            preorder_decision_tree(node->child[i], i, level+1);
        }
    }
}

int main(){
    std::vector<std::vector<int> > data;
    std::vector<int> class_value;
    std::vector<int> feature_name;
    int n;
    int m;
    int value;
    std::vector<Node *> tree;
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; i++){
        std::vector<int> temp;
        for (int j = 0; j < n; j++){
            scanf("%d", &value);
            temp.push_back(value);
        }
        data.push_back(temp);
        scanf("%d", &value);
        class_value.push_back(value);
    }
    for (int i = 0; i < n; i++){
        feature_name.push_back(i);
    }
    construct_desicion_tree(data, class_value, feature_name, tree);
    preorder_decision_tree(tree[0], 0, 0);
}
