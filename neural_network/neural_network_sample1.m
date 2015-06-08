
X = [1.000000, 0.084147, 0.090930;
	 1.00000, 0.090930, 0.065699;
	 1.000000, 2, 3];

W1 = [0.084147, -0.027942, -0.099999;
	  0.090930, 0.065699, -0.053657;
	  0.014112, 0.098936, 0.042017;
	  -0.075680, 0.041212, 0.099061];

W2 = [0.084147, -0.075680, 0.065699, -0.054402, 0.04201;
	  0.090930, -0.095892, 0.098936, -0.099999, 0.099061];

z2 = W1 * X';
a2 = [ones(1,size(z2,2)); sigmoid(z2)];
z3 = W2 * a2;
a3 = sigmoid(z3);
y = [1, 0, 0; 0 , 1, 1];
J = sum(sum(- y .* log(a3)- (1 - y) .* log(1 - a3))) / 3;

fprintf(['J = %f \n'], J);
fprintf(['a2:\n']);
disp(a2);
fprintf(['a3:\n']);
disp(a3);