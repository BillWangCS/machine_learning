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
        get_data_by_feature_value(data, class_value, feature_name,
								best_feature_name_pos, *it, new_data, new_class_value, new_feature_name);								
        node->child[*it] = construct_desicion_tree(new_data, new_class_value, new_feature_name, tree);		
    }	
    return node;
}

const char *g_feature_name[] = {
    "Deadline?",
    "Is there a dota game?",
    "Bad mood?"
};
const char *g_class_value[]= {
    "Dota",
    "Study",
    "Dating",
    "Basketball"
};

void preorder_decision_tree(Node *node, int value, int level){
    for (int i = 0; i < level; i++){
        printf("---");
    }
    if (level){
        printf("%d->", value);
    }    
    if (node->feature_name == -1) {
        printf("leaf(%s)\n", g_class_value[node->type]);
    }
    else{
        printf("inner(%s)\n", g_feature_name[node->feature_name]);
    }
    for (int i = 0; i < MAX_FEATURE_VALUE_NUMBER; i++){
        if (node->child[i]){
            preorder_decision_tree(node->child[i], i, level+1);
        }
    }
}

int predict(int sample[], Node *root){
    Node *ptr = root;
    while(ptr->type == -1){
        int value = sample[ptr->feature_name];
        ptr = ptr->child[value];
    }
    return ptr->type;
}

int main(){
    int data_ori [][10] = {
        {2, 1, 1}, {2, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 0, 1}, 
        {0, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {2, 0, 0}
    };
    int class_value_ori[] = {0, 1, 0, 0, 2, 0, 1, 3, 0, 1};
    std::vector<std::vector<int> > data;
    std::vector<int> class_value;
    std::vector<int> feature_name;
    for (int i = 0; i < 10; i++){
        std::vector<int> temp;
        for (int j = 0; j < 3; j++){
            temp.push_back(data_ori[i][j]);
        }
        data.push_back(temp);
		class_value.push_back(class_value_ori[i]);
    }
    for (int i = 0; i < 3; i++){
        feature_name.push_back(i);
    }
    std::vector<Node *> tree;
    construct_desicion_tree(data, class_value, feature_name, tree);
    preorder_decision_tree(tree[0], 0, 0);
    int sample1[] = {2, 1, 0};
    int sample2[] = {0, 0, 0};
    printf("sample1 predict = [%s]\n", g_class_value[predict(sample1, tree[0])]);
    printf("sample2 predict = [%s]\n", g_class_value[predict(sample2, tree[0])]);
    for (int i = 0; i < tree.size(); i++){
        delete tree[i];
    }
    return 0;
}

