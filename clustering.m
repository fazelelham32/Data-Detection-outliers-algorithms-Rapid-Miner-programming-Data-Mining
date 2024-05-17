clc;
clear;
close all;
c01=[0 0];
n1=20;
X1=randn(n1,2);
X1(:,1)=X1(:,1)+c01(1);
X1(:,2)=X1(:,2)+c01(2);

c02=[3 3];
n2=20;
X2=randn(n2,2);
X2(:,1)=X2(:,1)+c02(1);
X2(:,2)=X2(:,2)+c02(2);

c03=[4 2];
n3=20;
X3=randn(n3,2);
X3(:,1)=X3(:,1)+c03(1);
X3(:,2)=X3(:,2)+c03(2);

X=[X1, X2, X3];

% figure;
% plot(X1(:,1), X1(:,2), 'bo');
% hold on;
% plot(X2(:,1), X2(:,2), 'rs');
% plot(X3(:,1), X3(:,2), 'kp');
% axis equal;
% grid on;

figure;
plot(X1(:,1), X1(:,2),'bo');




