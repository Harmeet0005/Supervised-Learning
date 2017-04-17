clc;
clear;
%% 1
load('Train_All_Data_DigiLBP.mat');
load('Train_All_Label_DigiLBP.mat');
load('Test_All_Data_DigiLBP.mat');
A=(Train_All_Data_DigiLBP);
B=max(max(Train_All_Data_DigiLBP));
Train_All_Data_DigiLBP=A./B;
M=(Test_All_Data_DigiLBP);
N=max(max(Test_All_Data_DigiLBP));
Test_All_Data_DigiLBP=M./N;
%% 2
no_feat=size(Train_All_Data_DigiLBP,2);
accuracy=0;
Y=[];
%% 3 
x=load('Test_All_Label_DigiLBP.mat');
perf_val=zeros(no_feat);
for i=1:no_feat
    if ismember(i,Y)==0
        feat=[Y i];
        SVMModel = fitcsvm(Train_All_Data_DigiLBP(:,feat),Train_All_Label_DigiLBP);
        [label, score] = predict(SVMModel,Test_All_Data_DigiLBP(:,feat));
        test_lbp_label=x.Test_All_Label_DigiLBP;
        perf=classperf(test_lbp_label,label);
        perf_val(i)=perf.CorrectRate;
    end
end
[accuracy,argmax]=max(perf_val);
Y=[Y argmax];
%% 4
J=zeros;
for i=1:length(Y)
    feat=Y(Y~=Y(i));
    SVMModel = fitcsvm(Train_All_Data_DigiLBP(:,feat),Train_All_Label_DigiLBP);
    [label, score] = predict(SVMModel,Test_All_Data_DigiLBP(:,feat));
    test_lbp_label=x.Test_All_Label_DigiLBP;
    perf=classperf(test_lbp_label,label);
    J(i)=perf.CorrectRate ;
end

feat_drop=[]; %index for dropped features
feat_accu=[]; %values of the dropped features
for i=1:length(Y)
    if(J(i)>accuracy) %noting down the features whose values are higher than the previous best value
        feat_accu=[feat_accu J(i)];
        feat_drop=[feat_drop i];
    end
end
if isempty(feat_drop)==0 %removing the feature which would increase the performance
    [r,s]=max(feat_accu);
    accuracy=r;
    drop=feat_drop(s);
    Y=Y(Y~=Y(drop));
end
