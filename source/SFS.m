
function perf_value=SFS(no_feat,Y)

load('Train_All_Data_DigiLBP');
load('Train_All_Label_DigiLBP.mat');
load('Test_All_Data_DigiLBP.mat');
x=load('Test_All_Label_DigiLBP.mat');

perf_val=zeros(no_feat);

for i=1:no_feat
    if ismember(i,Y)==0
        feat=[Y i];
        SVMModel = fitcsvm(Train_All_Data_DigiLBP(:,feat),Train_All_Label_DigiLBP);
        [label, score] = predict(SVMModel,Test_All_Data_DigiLBP(:,feat));
        test_lbp_label=x.Test_All_Label_DigiLBP;
        perf=classperf(test_lbp_label,label);
        perf_value(i)=perf.CorrectRate;
    end
end
end