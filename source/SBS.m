
function J=SBS(Y)

load('Train_All_Data_DigiLBP');
load('Train_All_Label_DigiLBP.mat');
load('Test_All_Data_DigiLBP.mat');
x=load('Test_All_Label_DigiLBP.mat');

for i=1:length(Y)
   feat=Y(Y~=Y(i));
   %disp(feat);
   SVMModel = fitcsvm(Train_All_Data_DigiLBP(:,feat),Train_All_Label_DigiLBP);
        [label, score] = predict(SVMModel,Test_All_Data_DigiLBP(:,feat));
        test_lbp_label=x.Test_All_Label_DigiLBP;
        perf=classperf(test_lbp_label,label);
        J(i)=perf.CorrectRate ;
   
end
end
