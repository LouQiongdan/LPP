%���ǩK����ѧϰ����
%% ѵ���׶�
clear
load('.\data\TBFS\CAL500_TBFS\train_data.mat');
load('.\data\TBFS\CAL500_TBFS\train_target.mat');

%%%%%%%%%%%%%%%%%%%%%%%%����������һ���%%%%%%%%%%%%%%%%%%%%
% load('.\data\CAL500.mat');
% % test_target=test_target';%%%%%%������ֻ���image���ݼ��ر����
% % train_target=train_target';%%%%%%������ֻ���image���ݼ��ر����
% target=target';
% train_data=data(1:402,:);
% train_target=target(1:402,:);
% test_data=data(403:end,:);
% test_target=target(403:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%����������һ���%%%%%%%%%%%%%%%%%%%%

train_target=train_target';%һ����һ������


Num=10;
Smooth=0.01;%��Ҫ֪��k���ڵĽ���������ƽ������
[train_data,~] = mapminmax(train_data,0,1); %��һ������
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
%% ���Խ׶�
load('.\data\TBFS\CAL500_TBFS\test_data.mat');
load('.\data\TBFS\CAL500_TBFS\test_target.mat');

test_target=test_target';%һ����һ������

[test_data,~] = mapminmax(test_data,0,1); %��һ������
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);

