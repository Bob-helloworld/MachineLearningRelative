 clear all
 clc
 
%  mex cec14_func.cpp -DWINDOWS
func_num=1;
D=30; %测试函数维度
% Xmin=-10; %测试x解空间
% Xmax=10;
Xmin=[-100 -5.12 -32 -10 0 -500 -5.12 -5.12 -65.536 -100 -1 -5 -5 -10]; %针对特定函数，不同的上下边界
Xmax=[100 5.12 32 10 10 500 5.12 5.12 65.536 100 1 10 10 10];
%特别注意，在修改func.m中函数顺序之后，如果不是通用的定义域，一定修改定义域

pop_size=20; %种群数
iter_max=1e4*D; %最大迭代次数
runs=5; %重复次数
funcnum=3; %测试到第几个函数
fhd=str2func('cec13_func'); 
opt1=[-1400:100:-100];
opt2=[100:100:1400];
fopt=[opt1,opt2]; %目标函数的最优值数组拼接，使用CEC13的时候用,注意在使用普通函数的时候ERR是无用的
Algstr={'MQA_SS05','BBFWA','CLPSO','QPSO','SPSO2011','mqhoaAs','mqhoaA'};
Algnum=6;  %选取使用的测试算法(上表)
namestr=['.\data\',Algstr{Algnum},['_cec14_result_D',num2str(D),'_P',num2str(pop_size),'_',datestr(now, 'yyyymmddHHMMSS')]];

txtstr=[namestr,'.txt'];
fp=fopen(txtstr,'w+');
for i=2:funcnum    %选择测试函数，func.m中有14个高维函数，单独测试，写成funcnum:funcnum
    func_num=i;  %i为调用函数ID
%     fprintf('%s \n',datestr(now, 'yyyy/mm/dd|HH:MM:SS'));
%     fprintf(fp,'%s \r\n',datestr(now, 'yyyy/mm/dd|HH:MM:SS'));
    for j=1:runs %j为设置的循环次数
        tic;
            switch(Algnum) %现在可用的算法有SPSO2011 MQHOAS_ex mqhoaA QPSO
                case 1
                    [gbest,gbestval,FES]= mqa_ss_comp(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
                case 2
                    [gbest,gbestval,FES]= bbfwa(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
                case 3
                    [gbest,gbestval,FES]= CLPSO(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
                case 4
                    [gbest,gbestval,FES]= QPSO_func(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
                case 5
                    [gbest,gbestval,FES]= SPSO2011(pop_size,D,iter_max,Xmin(i),Xmax(i),func_num);
                case 6
                    [gbest,gbestval,FES]= mqhoaAs(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
                case 7
                    [gbest,gbestval,FES]= mqhoaA(D,pop_size,iter_max,Xmin(i),Xmax(i),func_num);
            end
            
        tot_time(j) = toc;
        xbest(i,:)=gbest;
        fbest(i,j)=gbestval;
        Err(i,j)=fbest(i,j)-fopt(i);        
        FESM(i,j)=FES;
        fprintf('Fuc= f%d,  No. %d run, DIM=%d, Global minimum=%e. FE=%d, time=%d\n',i,j,D,fbest(i,j),FES,toc);
        fprintf('Xmin=%d,  Xmax= %d \n',Xmin(i),Xmax(i)); %显示当前边界
        %fprintf(fp,'Fuc= f%d,  No. %d run, DIM=%d, Global minimum=%e. FE=%d, time=%d\r\n',i,j,D,fbest(i,j),FES,toc);
    end %测试算法迭代结束
   %输出至屏幕显示
    fprintf('---------------------------------------------------------------\n');        
   fprintf('Repeat=%d, Mean FE=%1.2e,Meantime=%1.2e\n',runs,mean(FES),mean(tot_time));        
   fprintf('MeanValue=%1.2e, BestValue=%1.2e, Std=%1.2e, \n',mean(fbest(i,:)),min(fbest(i,:)),std(fbest(i,:)));        
   fprintf('MeanErr=%1.2e, BestErr=%1.2e, StdErr=%1.2e, \n',mean(Err(i,:)),min(Err(i,:)),std(Err(i,:)));
   fprintf('%s \n',datestr(now, 'yyyy/mm/dd|HH:MM:SS'));
   %输出至txt文件
%    fprintf(fp,'---------------------------------------------------------------\r\n');
%    fprintf(fp,'Repeat=%d, Mean FE=%1.2e,Meantime=%1.2e\r\n',runs,mean(FES),mean(tot_time));        
%    fprintf(fp,'MeanValue=%1.2e, BestValue=%1.2e, Std=%1.2e, \r\n',mean(fbest(i,:)),min(fbest(i,:)),std(fbest(i,:)));        
%    fprintf(fp,'MeanErr=%1.2e, BestErr=%1.2e, StdErr=%1.2e, \r\n',mean(Err(i,:)),min(Err(i,:)),std(Err(i,:)));
%    fprintf(fp,'%s \r\n',datestr(now, 'yyyy/mm/dd|HH:MM:SS'));  
   
end %测试函数迭代结束
%最优值处理
for i=1:funcnum
fbestall(i)=min(fbest(i,:));
end 
%均值处理
for i=1:funcnum
fmeanall(i)=mean(fbest(i,:));
end 
%方差处理
for i=1:funcnum
fstdall(i)=std(fbest(i,:));
end 
%成功率处理
for i=1:funcnum
    SRcount=0;
    for j=1:runs
        if(fbest(i,j)<(1E-3))
        SRcount=SRcount+1;
        end
    end
    SRrate(i)=SRcount/runs*100;
end 

% xlsxstr=[namestr,'.xlsx'];
% xlswrite(xlsxstr,xbest,'xbest');
% xlswrite(xlsxstr,fbest,'fbest');
% xlswrite(xlsxstr,FESM,'FESM');
% xlswrite(xlsxstr,Err,'Err');
% 
% xlswrite(xlsxstr,fbestall,'fbestall');
% xlswrite(xlsxstr,fmeanall,'fmeanall');
% xlswrite(xlsxstr,fstdall,'fstdall');
% xlswrite(xlsxstr,SRrate,'SRrate');
% %%%%%%%%%%%拼接数据%%%%%%%%%%
% staterr=[fbestall;fmeanall;fstdall;SRrate];
% xlswrite(xlsxstr,staterr,'staterr');
% fclose(fp);
