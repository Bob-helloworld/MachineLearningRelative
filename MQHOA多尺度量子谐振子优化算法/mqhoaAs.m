%------------------------------------------------------------------
% ���ܼ��ȶ��汾
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
%--������������ʼ------------------------------
%repeat          �ظ��������
%DIM             Ŀ�꺯��ά��
%sigmaMin        ��С�߶�
%groupNum        ��˹����������Ŀ����kֵ
%minDomain       Ŀ�꺯���������½�
%minDomain       Ŀ�꺯���������Ͻ�
%-------------------------------------------------------%
repeat=1;
DIM=Dimension;
sigmaMin=0.000001; 
groupNum=Particle_Number;
minDomain=VRmin; maxDomain=VRmax;
maxFE=Max_Gen;

%------------------------------------------------------%

%------����ظ������������------
%gbestV          �洢ÿ�μ�������ź���ֵ
%gfe             �洢ÿ�μ���ĺ�����������            
%tot_time        �洢ÿ�μ����ʱ��
%-----------------------------------------------------%
gbestV=zeros(1,repeat);  
gfe=zeros(1,repeat); 
tot_time = zeros(1,repeat);
%-----------------------------------------------------%

for rep=1:repeat
    tic;
%--��ʼ������ʼ--------------------------------------%
%funcV             �洢k��������ĺ���ֵ��������
%samplePos         �洢��ǰk�����������꣨����
%optimalSolution   �洢��ǰk�����Ž�����꣨����
%sigma             ��ǰ�߶ȣ�������
%stdNow            ��ǰÿ��ά���ϲ�����ı�׼�������
%stdPre            ��һ��ÿ��ά���ϲ�����ı�׼�������
%w                 ����������������(����)
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
    while (evolotiontime<maxFE) % �߶ȵ�����ʼ 
        while (evolotiontime<maxFE) %г���ӵ�����ʼ
            while (evolotiontime<maxFE)  %�ܼ��ȶ�����������ʼ
                change_flag=0; % op_solution�����жϱ�־
                for k=1:groupNum
%--����Box-Muller��������DIMά�µ���̬�ֲ�������
%--------------------------------------------%
                    theat=2*pi*rand(DIM,1);
                    R=sqrt(-2.0*log(rand(DIM,1))); 
                    gaosiRand=R.*cos(theat);  
                    samplePos(:,k)=optimalSolution(:,k)+sigma*gaosiRand;
%---------------------------------------------%
  
%--����Խ��Ĳ�����------------------%
%--����Խ��������������߽��������%
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
%--�Ա����β���֮�亯��ֵ�Ĵ�С�����滻���---------%
%----------------------------------------------------%
                     sampleValue=func(samplePos(:,k),DIM,varargin{:}); %funcV(k)=func(optimalSolution(:,k),DIM,varargin{:})
                     evolotiontime=evolotiontime+1; %��������������¼
                     if sampleValue<funcV(k)     
                        funcV(k)=sampleValue;
                        optimalSolution(:,k)=samplePos(:,k); 
                        change_flag=1;
                     end
%-------------------------------------------------------%
                end
%--�ܼ��ȶ��о�----------------------
%--------------------------------------------------%        
%                 �Ա�־λchange_flag�����жϣ���ѯ�ܼ��ȶ��������Ƿ��op_solution���й��޸�
                  if(change_flag==0)
                  break; 
                  end
%--------------------------------------------------%
            end %�ܼ��ȶ���������
                          
%--�ܼ��½���������ֵ�滻------------------------------%
%------------------------------------------------------%
           meanPos=mean(optimalSolution,2);%ȡ��ƽ������
           [v_max,index_max]=max(funcV);%ȡ�����ֵ�����index_max
           optimalSolution(:,index_max)=meanPos;%��ƽ�������滻���ֵ��Ӧ����          
           funcV(index_max)=func(meanPos,DIM,varargin{:}); %funcV(k)=func(optimalSolution(:,k),DIM,varargin{:})
           evolotiontime=evolotiontime+1;
           stdPre=std(optimalSolution,1,2);  %�½��׼��
%------------------------------------------------------%

%------��Сֵ�滻
%             [v_min,index_min]=min(funcV);
%             minPos=optimalSolution(:,index_min);%ȡ��ƽ������
%            [v_max,index_max]=max(funcV);%ȡ�����ֵ�����index_max
%            optimalSolution(:,index_max)=minPos;%��ƽ�������滻���ֵ��Ӧ����          
%            funcV(index_max)=func(minPos,DIM);
%            w=w+1;
%            stdPre=std(optimalSolution,1,2);  %�½��׼��     
%------��Сֵ�滻     

           if max(stdPre)<sigma%�߶��½��о�
               break;
           end
        end %г���ӵ�������  
       sigma=sigma/2.0;%�߶��½�����
       if sigma<=sigmaMin %�����о� 
           break;
       end
    end % �߶ȵ�������
%--��¼���μ���Ľ������ӡ----------------------
%------------------------------------------
    tot_time(rep) = toc; 
    [global_min,index]=min(funcV);
    gbestV(rep)=min(funcV);
    gfe(rep)=evolotiontime;
%    fprintf('No. %d run, DIM=%d, Global minimum=%e. FE=%d, time=%d\n',rep,DIM,global_min,w,toc);
%------------------------------------------------
end % �ظ�����
    fitcount=evolotiontime;
    gbestval=global_min;
    gbest=(optimalSolution(:,index))'; %��֮ǰfunV������ֵ�����ȡ��������ת�󣬴�������