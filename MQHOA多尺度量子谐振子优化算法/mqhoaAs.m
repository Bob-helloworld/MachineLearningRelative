%------------------------------------------------------------------
% 有能级稳定版本
% MQHOA algorithm for sigle model objective function
% Matlab Code by Wang Peng (May. 07, 2017).
% This program must include func.m
% This is global minimum program.
% All rights reserved by Parallel Computing Lab.
%------------------------------------------------------------------
function [gbest,gbestval,fitcount]= mqhoaAs(Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
%clear all;
feature jit off;
format long;
%global DIM;
%--参数定义区域开始------------------------------
%repeat          重复计算次数
%DIM             目标函数维度
%sigmaMin        最小尺度
%groupNum        高斯采样区域数目，即k值
%minDomain       目标函数定义域下界
%minDomain       目标函数定义域上界
%-------------------------------------------------------%
repeat=1;
DIM=Dimension;
sigmaMin=0.000001; 
groupNum=Particle_Number;
minDomain=VRmin; maxDomain=VRmax;
maxFE=Max_Gen;

%------------------------------------------------------%

%------多次重复计算所需变量------
%gbestV          存储每次计算的最优函数值
%gfe             存储每次计算的函数进化次数            
%tot_time        存储每次计算的时间
%-----------------------------------------------------%
gbestV=zeros(1,repeat);  
gfe=zeros(1,repeat); 
tot_time = zeros(1,repeat);
%-----------------------------------------------------%

for rep=1:repeat
    tic;
%--初始化区域开始--------------------------------------%
%funcV             存储k个采样点的函数值（向量）
%samplePos         存储当前k个采样点坐标（矩阵）
%optimalSolution   存储当前k个最优解的坐标（矩阵）
%sigma             当前尺度（标量）
%stdNow            当前每个维度上采样点的标准差（向量）
%stdPre            上一次每个维度上采样点的标准差（向量）
%w                 函数进化次数计数(标量)
%-------------------------------------------------------%
    funcV=zeros(1,groupNum);
    samplePos=zeros(DIM,groupNum);
    sigma=maxDomain-minDomain;
    stdPre=zeros(DIM,1);
    stdNow=zeros(DIM,1);
    optimalSolution=unifrnd(minDomain,maxDomain,DIM,groupNum);
    stdPre=std(optimalSolution,1,2) ;
    evolotiontime=0;
    for k=1:groupNum 
        funcV(k)=func(optimalSolution(:,k),DIM,varargin{:}); %funcV(k)=func46(optimalSolution(:,k),DIM,varargin{:})
        evolotiontime=evolotiontime+1;  
    end
%---------------------------------------------------%
    while (evolotiontime<maxFE) % 尺度迭代开始 
        while (evolotiontime<maxFE) %谐振子迭代开始
            while (evolotiontime<maxFE)  %能级稳定收敛迭代开始
                change_flag=0; % op_solution更新判断标志
                for k=1:groupNum
%--采用Box-Muller方法生成DIM维新的正态分布采样点
%--------------------------------------------%
                    theat=2*pi*rand(DIM,1);
                    R=sqrt(-2.0*log(rand(DIM,1))); 
                    gaosiRand=R.*cos(theat);  
                    samplePos(:,k)=optimalSolution(:,k)+sigma*gaosiRand;
%---------------------------------------------%
  
%--处理越界的采样点------------------%
%--对于越界采样点采用最近边界坐标替代%
%------------------------------------------------%
                    for d=1:DIM
                        if samplePos(d,k)>maxDomain
                            samplePos(d,k)=maxDomain;                 
                        end
                        if samplePos(d,k)<minDomain
                            samplePos(d,k)=minDomain;  
                        end
                    end
%-------------------------------------------------%
%--对比两次采样之间函数值的大小，并替换差解---------%
%----------------------------------------------------%
                     sampleValue=func(samplePos(:,k),DIM,varargin{:}); %funcV(k)=func(optimalSolution(:,k),DIM,varargin{:})
                     evolotiontime=evolotiontime+1; %函数进化次数记录
                     if sampleValue<funcV(k)     
                        funcV(k)=sampleValue;
                        optimalSolution(:,k)=samplePos(:,k); 
                        change_flag=1;
                     end
%-------------------------------------------------------%
                end
%--能级稳定判据----------------------
%--------------------------------------------------%        
%                 对标志位change_flag进行判断，查询能级稳定过程中是否对op_solution进行过修改
                  if(change_flag==0)
                  break; 
                  end
%--------------------------------------------------%
            end %能级稳定迭代结束
                          
%--能级下降操作，均值替换------------------------------%
%------------------------------------------------------%
           meanPos=mean(optimalSolution,2);%取得平均坐标
           [v_max,index_max]=max(funcV);%取得最大值的序号index_max
           optimalSolution(:,index_max)=meanPos;%用平均坐标替换最大值对应坐标          
           funcV(index_max)=func(meanPos,DIM,varargin{:}); %funcV(k)=func(optimalSolution(:,k),DIM,varargin{:})
           evolotiontime=evolotiontime+1;
           stdPre=std(optimalSolution,1,2);  %新解标准差
%------------------------------------------------------%

%------最小值替换
%             [v_min,index_min]=min(funcV);
%             minPos=optimalSolution(:,index_min);%取得平均坐标
%            [v_max,index_max]=max(funcV);%取得最大值的序号index_max
%            optimalSolution(:,index_max)=minPos;%用平均坐标替换最大值对应坐标          
%            funcV(index_max)=func(minPos,DIM);
%            w=w+1;
%            stdPre=std(optimalSolution,1,2);  %新解标准差     
%------最小值替换     

           if max(stdPre)<sigma%尺度下降判据
               break;
           end
        end %谐振子迭代结束  
       sigma=sigma/2.0;%尺度下降操作
       if sigma<=sigmaMin %精度判据 
           break;
       end
    end % 尺度迭代结束
%--记录本次计算的结果并打印----------------------
%------------------------------------------
    tot_time(rep) = toc; 
    [global_min,index]=min(funcV);
    gbestV(rep)=min(funcV);
    gfe(rep)=evolotiontime;
%    fprintf('No. %d run, DIM=%d, Global minimum=%e. FE=%d, time=%d\n',rep,DIM,global_min,w,toc);
%------------------------------------------------
end % 重复计算
    fitcount=evolotiontime;
    gbestval=global_min;
    gbest=(optimalSolution(:,index))'; %把之前funV里最后的值的序号取出来，旋转后，存入数组