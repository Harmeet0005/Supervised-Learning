clc;
clear;
%% Loading All Training Data Set 
load('Train_All_Data_DigiLBP');
load('Train_All_Label_DigiLBP.mat');
load('Test_All_Data_DigiLBP.mat');
x=load('Test_All_Label_DigiLBP.mat');
%% Initialization of Matrix Elements
no_feat=size(Train_All_Data_DigiLBP,2);
Y=[];  
argvalue=0;
%% Implementation of SFS Algorithm 
perf_val=SFS(no_feat,Y);
[argvalue,argmax]=max(perf_val);
Y=[Y argmax];
%% Implementation of SBS Algorithm(for backtracking)
J=SBS(Y); %backtracking
            feat_drop=[]; %index for dropped features
            feat_accu=[]; %values of the dropped features
            for i=1:length(Y)
                if(J(i)>argvalue) %noting down the features whose values are higher than the previous best value
                    feat_accu=[feat_accu J(i)];
                    feat_drop=[feat_drop i];
                end
            end
            if isempty(feat_drop)==0 %removing the feature which would increase the performance
                [r,s]=max(feat_accu);
                argvalue=r;
                drop=feat_drop(s);
                Y=Y(Y~=Y(drop));
            end

