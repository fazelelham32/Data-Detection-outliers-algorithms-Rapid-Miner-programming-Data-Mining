clc;
clear;
close all;

M = [0 0
     3 4
     6 1];

k=size(M,1);

N = 100;

sigma = 1;

X = [];
for j=1:k
    
    mj=M(j,:);
    
    Xj = mvnrnd(mj, sigma*ones(size(mj)), N);
    
    X=[X
       Xj];
end

plot(X(:,1),X(:,2),'x');

save('mydata', 'X');

