 load('D:\Lqd_CX\日常降维学习算法\自己写的\LPP_lqd\data\haberman.mat');
 Data = data(:,1:3);
 Target = data(:,4);
 
 train_target=Target(1:245,:);
 test_target=Target(246:end,:);
 train_target=train_target';
 test_target=test_target';
 
 
 new_dim = 1;
 Num=10;
 Smooth=0.01;
 
 [ EigVec,EigVal] = LPP_by_lqd( Data' ,new_dim);
 New_data=Data*EigVec;
 
 disp('-------------------------这是不经过降维的效果-------------------------------------');
 train_data1=Data(1:245,:);
 test_data1=Data(246:end,:);
 [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data1,train_target,Num,Smooth);
 [HammingLoss1,~,~,~,~,~,~]=MLKNN_test(train_data1,train_target,test_data1,test_target,Num,Prior,PriorN,Cond,CondN);
 
 disp('-------------------------------这是经过LPP降维之后的效果--------------------------------------------------');
 train_data2=New_data(1:245,:);
 test_data2=New_data(246:end,:);
 [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data2,train_target,Num,Smooth);
 [HammingLoss2,~,~,~,~,~,~]=MLKNN_test(train_data2,train_target,test_data2,test_target,Num,Prior,PriorN,Cond,CondN);
 
 