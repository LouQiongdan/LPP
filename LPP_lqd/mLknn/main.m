%多标签K近邻学习方法
%% 训练阶段
clear
load('.\data\TBFS\CAL500_TBFS\train_data.mat');
load('.\data\TBFS\CAL500_TBFS\train_target.mat');

%%%%%%%%%%%%%%%%%%%%%%%%以下五行是一起的%%%%%%%%%%%%%%%%%%%%
% load('.\data\CAL500.mat');
% % test_target=test_target';%%%%%%这两行只针对image数据集特别添加
% % train_target=train_target';%%%%%%这两行只针对image数据集特别添加
% target=target';
% train_data=data(1:402,:);
% train_target=target(1:402,:);
% test_data=data(403:end,:);
% test_target=target(403:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%以上五行是一起的%%%%%%%%%%%%%%%%%%%%

train_target=train_target';%一列是一个样本


Num=10;
Smooth=0.01;%需要知道k近邻的紧邻数量和平滑参数
[train_data,~] = mapminmax(train_data,0,1); %归一化处理？
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
%% 测试阶段
load('.\data\TBFS\CAL500_TBFS\test_data.mat');
load('.\data\TBFS\CAL500_TBFS\test_target.mat');

test_target=test_target';%一列是一个样本

[test_data,~] = mapminmax(test_data,0,1); %归一化处理？
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);

