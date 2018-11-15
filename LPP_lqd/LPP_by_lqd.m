function [ EigVec,EigVal] = LPP_by_lqd( data ,new_dim)
%UNTITLED 此处显示有关此函数的摘要
%Input:
    % data :the original datasets ,whose dimension is d*N.d is the number of features and N is the number of examples.
    % new_dim:the goal lower dimension of datasets after the process of LPP 
%Output:
    % EigVec :the first new_dim EigVectors according to EigVal
    % EigVal:the first new_dim smallest EigValue.
%   此处显示详细说明
[d,N]=size(data);
W=zeros(N,N);
for i = 1:N
    for j = 1:N 
        vec_a=data(:,i);
        vec_b=data(:,j);
        W(i,j)=dot(vec_a,vec_b)/(norm(vec_a)*norm(vec_b));
    end
end
D=zeros(N,N);
L=zeros(N,N);
for i = 1: N 
    D(i,i)=sum(W(i,:),2);
end
L=D-W;
[eigvec,eigval]=eig(data*L*data',data*D*data');
tem_eigval=diag(eigval);
[K,index]=sort(tem_eigval);
EigVal=tem_eigval(index(1:new_dim),:);
EigVec=eigvec(:,index(1:new_dim));
end

