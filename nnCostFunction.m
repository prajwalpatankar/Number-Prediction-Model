function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Variables to return
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m,1) X];

z2 = a1 * Theta1';

a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];

z3 = a2 * Theta2';

a3 = sigmoid(z3);

hofx = a3;
%%%%%% get h of x


y_Vec = zeros(m,num_labels);

for i=1:m
    y_Vec(i,y(i)) = 1;  %y_Vec[i][ y[i] ] = 1 basically
endfor
%%%%%% get y vector


J = ( (1/m) * sum( sum (( -y_Vec .* log(hofx) ) - ( ( 1 - y_Vec).* log(1-hofx) ) ) ) )  +  ( (lambda/(2*m)) * ( ( sum( sum( Theta1(:,2:size(Theta1,2) ) .^2 ) ) )  + ( sum( sum( Theta2(:,2:size(Theta2,2)).^2  ) ) ) ) ) ;
%%%%%% cost func with regularisation


delta3 = a3 - y_Vec;

delta2 = (delta3 * Theta2) .* [ones( size(z2,1) , 1 ) sigmoidGradient(z2) ];

delta2 = delta2(:,2:end);

Theta1_grad = (1/m) * ( delta2' * a1 );

Theta2_grad = (1/m) * ( delta3' * a2 );
%%%%%% backprop


Theta1_grad = (1/m) * ( delta2' * a1 )  +  (lambda/m) *[ zeros( size(Theta1,1),1) Theta1(:,2:end) ] ;

Theta2_grad = (1/m) * ( delta3' * a2 )  +  (lambda/m) *[ zeros( size(Theta2,1),1) Theta2(:,2:end) ] ;
%%%%%% backprop with regularisation



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
