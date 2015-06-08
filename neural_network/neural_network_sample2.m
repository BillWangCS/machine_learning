
X = [1.000000, 0.084147, 0.090930;
	 1.00000, 0.090930, 0.065699;
	 1.000000, 2, 3];

W1 = [0.084147, -0.027942, -0.099999;
	  0.090930, 0.065699, -0.053657;
	  0.014112, 0.098936, 0.042017;
	  -0.075680, 0.041212, 0.099061];

W2 = [0.084147, -0.075680, 0.065699, -0.054402, 0.04201;
	  0.090930, -0.095892, 0.098936, -0.099999, 0.099061];

y = [1; 2; 2];

grad1_sum = 0;
grad2_sum = 0;

m = 3;

for i = 1:m
	a1 = X(i,:)';
	z2 = W1 * a1;
	a2 = [1 ; sigmoid(z2)];
	z3 = W2 * a2;
	a3 = sigmoid(z3);
	yi = zeros(2, 1);
	yi(y(i)) = 1;
	delta3 = a3-yi;
	delta2 = W2'*delta3.*sigmoidGradient([1;z2]);
	grad2_sum += delta3 * a2';
	grad1_sum += delta2(2:end) * a1';
end

grad1_sum = grad1_sum/m;
grad2_sum = grad2_sum/m;

fprintf(['w1_grad:\n']);
disp(grad1_sum);
fprintf(['w2_grad:\n']);
disp(grad2_sum);
