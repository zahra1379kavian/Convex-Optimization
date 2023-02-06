close all
clc
clear

%%%%%Dual SVM Problem
A = importdata('svm_train.txt');
X = A(:,1:2);
y = A(:,3);
Xt = diag(y)*X;
n = size(X,1);
gamma = [10,50,100,500];
C = [0.01,0.1,0.5,1];
%gamma  = 100;
%C = 0.5;
%%

for t = 1:length(gamma)
    for k = 1:length(C)
Q = zeros(n,n);

for i = 1:n
    for j = 1:n
    Q(i,j) = exp(-gamma(1,t)*norm(X(i,:)-X(j,:))^2);   
    end
end

cvx_begin
variables alph(n)
minimize (0.5.*quad_form(y.*alph,Q) - ones(n,1)'*alph)
subject to 
        y'*alph == 0;
        alph >= 0;
        alph <= C(1,k);
cvx_end
optimal_value(t,k) = cvx_optval;
optimal_alpha(t,k,:) = alph;
    end
end
%%
optimal_value = cell2mat(struct2cell(load('optimal_value.mat')));
optimal_alpha = cell2mat(struct2cell(load('optimal_alpha.mat')));

%%%%Find other parameter
%W1 = sum(alph.*y.*X(:,1),'all');
%W2 = sum(alph.*y.*X(:,2),'all');
%W = [W1 W2];

%b = mean(y'-W*X','all');
%%
%%%decision border
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)]; 

%%
for t = 1:length(gamma)
    for k = 1:length(C)
    Alpha = squeeze(optimal_alpha(t,k,:));
        
    T = X(Alpha>1e-5,:);
    Ty = y(Alpha>1e-5);
    Talpha = Alpha(Alpha>= 1e-5);

    X1 = sum(xGrid.^2, 2);
    X2 = sum(T.^2, 2)';
    K1 = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * xGrid * T'));
    K2 = gaussianKernel(1, 0,C(1,k)) .^ K1;
    K3 = bsxfun(@times, Ty', K2);
    K = bsxfun(@times, Talpha', K3);
    p = sum(K, 2);
    scores(t,k,:) = p;
    end
end

%%
clc
for t = 1:length(gamma)
    for k = 1:length(C)
        figure
        gscatter(X(:,1),X(:,2),y)
        hold on
        s = (scores(t,k,:));
        z = reshape(s,size(x1Grid,1),[]);
        contour(x1Grid,x2Grid,z);
        hold on
        plot(T(:,1),T(:,2),'bo');
        title(['gamma : ',num2str(gamma(1,t)),' C : ',num2str(C(1,k))]);
        hold off
    end
end

%%
%%%%supporting vectors
for t = 1:length(gamma)
    for k = 1:length(C)
    Alpha = squeeze(optimal_alpha(t,k,:));
        
    sp(t,k) = size(X(Alpha>1e-5,:),1);
    end
end

%%
%%%%Barrier Methode
clc

gamma = 500;
C = 0.1;
mu = 10e-4;

r = 2.*ones(n,1);
while r'*y == 0
r = C.*rand(n,1);%initial point
end

alph = r;

Q = zeros(n,n);
grad2 = zeros(n,n);

t = 10;
s = 0.01;
beta = 0.5;
%%

while mu>0.5e-8

for i = 1:n
    for j = 1:n
    Q(i,j) = exp(-gamma*norm(X(i,:)-X(j,:))^2);   
    end
end


for i = 1:n
    for j = 1:n
       grad2(i,j) = y(i,1)*y(j,1)*Q(i,j);        
    end
end

grad1 = ones(n,1)-grad2*alph + mu.*(ones(n,1)'*(1./log(alph))-ones(n,1)'*(1./log(C-alph)));
grad2 = -grad2 + mu.*(ones(n,1)'*(-1./(log(alph)).^2)-ones(n,1)'*(1./log(C-alph).^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Netown Methode

A =  pinv([grad2 y;y' 0])*[-grad1;0];
deltax = A(1:n,1);
landa = A(n+1,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = alph;
x1 = x0 + t*deltax;

drawnow
AAAA = 0.5*quad_form(y.*x0,Q) - ones(n,1)'*x0-mu*(log(x0)+log(x0-C));
plot(alph,0.5*quad_form(y.*x0,Q) - ones(n,1)'*x0-mu*(log(x0)+log(x0-C)))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%backtracking

t = 10;
while (0.5*quad_form(y.*x1,Q) - ones(n,1)'*x1-mu*(log(x1)+log(x1-C))) > (0.5.*quad_form(y.*x0,Q) ...
    - ones(n,1)'*x0-mu*(log(x0)+log(x0-C)))+s.*t.*grad1'*deltax
    x0 = x1;
    t = t*beta;
    x1 = x0 + t*deltax;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alph = alph + t.*deltax;

mu = mu*0.5;
end

%%
sopport_value = size(X(Alpha>1e-5,:),1)




















