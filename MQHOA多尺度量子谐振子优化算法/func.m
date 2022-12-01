function y = func(x,DIM,funcNO) 
switch (funcNO)
%撇掉了双阱函数，并取消掉了相应的3个参数
%   case 0
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % double well function% The global minima: f(x)=xx
% % % % % Position (unknown)
% % % % % DIM=1
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n=DIM;
% sum=0;
% V = V_in;%6
% a =a_in;% 2
% delta = delta_in;% 0.3
%  for DW_i=1:n    %计算当前的双井值
%     sum = sum+ V*((x(DW_i)^2-a^2)^2)/(a^4)+delta*x(DW_i);
%  end
%  y=sum; 
% many local minia    
    case 1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Griewank function% The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)

% n=DIM;
% sum=0;
% V = V_in;%6
% a =a_in;% 2
% delta = delta_in;% 0.3
%  for DW_i=1:n    %计算当前的双井值
%     sum = sum+ V*((x(DW_i)^2-a^2)^2)/(a^4)+delta*x(DW_i);
%  end
%  y=sum; 

n = DIM;
s=0;k=1;
for i=1:n
	s=(x(i)).^2+s;
	k=cos((x(i))/sqrt(i))*k;
end
	s=s/4000;
	y=s-k+1;


    case 2
        
% %%%%%%%%%%%%%%%%%%%
% Rastrigin function
% The global minima: f(x)=0
% Position (0)
% %%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM; 
s = 0;
for j = 1:n
    s = s+(x(j)^2-10*cos(2*pi*x(j))); 
end
y = 10*n+s;

    case 3
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Ackley function.
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
a = 20; b = 0.2; c = 2*pi;
s1 = 0; s2 = 0;
for i=1:n
   s1 = s1+x(i)^2;
   s2 = s2+cos(c*x(i));
end
y = -a*exp(-b*sqrt(1/n*s1))-exp(1/n*s2)+a+exp(1);

    case 4
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Levy function.
% % The global minima: f(x)=0
% % Position (1)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
for i = 1:n
    z(i) = 1+(x(i)-1)/4; 
end
s = sin(pi*z(1))^2;
for i = 1:n-1
    s = s+(z(i)-1)^2*(1+10*(sin(pi*z(i)+1))^2);
end 
y = s+(z(n)-1)^2*(1+(sin(2*pi*z(n)))^2);

    case 5
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Alpine function
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM; 
s=0;
 for i = 1:n
        s = s + abs(x(i)*sin(x(i)) + 0.1*x(i));        
 end
 y=s;
%  
    case 6
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Schwefel function
% % The global minima: f(x)=0
% % Position (420.9687)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function y = func(x,DIM)
n = DIM;
s=0;
for i=1:n
	m=sin(sqrt(abs(x(i))));
	s=x(i)*m+s;
end
	y=418.9829*n-s;
    
    case 7
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Sphere function 
% % The global minima: f(x)=0
% % Position (0)
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
s = 0;
for j = 1:n
    s = s+x(j)^2; 
end
y = s;

    case 8
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Sum Squares Function 
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
s = 0;
for j = 1:n  
    s=s+j*x(j)^2; 
end
y = s;

    case 9
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5 Rotated Hyper-ellipsoid  nD xi=[-65.536,65.536] with f(0,...,0)=0 
% runs well when n=50
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = DIM; 
s=0;y=0;
for i=1:n
	for j=1:n
		s=s+x(j)^2;
	end
	y=y+s;
end	

    case 10
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6 Ellipsoidal --nD---with f(1,2,...,n)=0 
% when D>= 11 the algorithm fails 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = DIM; 
s=0;
for i=1:n 
	s=(x(i)-i).^2+s;
end
	y=s;
  
    case 11   
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7 Sum of different power nD -xi=[-1,1]  with f(0,...,0)=0 ========================OK
% slow down when n>=4 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = DIM; 
s=0;
for i=1:n
	s=abs(x(i))^(i+1)+s;
end	
	y=s;

    case 12
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Zakharov function.
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
s1 = 0;
s2 = 0;
for j = 1:n
    s1 = s1+x(j)^2;
    s2 = s2+0.5*j*x(j);
end
y = s1+s2^2+s2^4;

    case 13
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Rosenbrock function
% % The global minima: f(x)=0
% % Position (1)
% % DIM>1
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
sum = 0;
for j = 1:n-1
    sum = sum+100*(x(j)^2-x(j+1))^2+(x(j)-1)^2;
end
y = sum;

    case 14
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Dixon and Price function.
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
n = DIM;
s1 = 0;
for j = 2:n
    s1 = s1+j*(2*x(j)^2-x(j-1))^2;    
end
y = s1+(x(1)-1)^2;
% 
% 
%Quadric 存在问题，不能使用
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Quadric function
% % The global minima: f(x)=0
% % Position (0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = func(x,DIM)
% n = DIM; 
% s=0;
% s1 = 0;
%  for i = 1:DIM
%         for j=1:i
%              s1 = s1+x(j); 
%         end
%         s=s+s1^2;
%         s1=0;
%  end
%  y=s;
%  


    
%     case 16
% % 5 Styblinski-Tang   nD --xi=[-5,5] with f(-2.903534,...,-2.903534)=-39.16599*n =========OK
% % wrong results when D>=12
% n = DIM; 
% s=0;
% for i=1:n
% 	s=x(i)^4-16*x(i)^2+5*x(i)+s;
% end	
% 	y=0.5*s;

% @@@@@@@@@@@@@@@@@@@以下是2维测试函数@@@@@@@@@@@@@@
%     case 13
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Drop-Wave function
% % % The global minima: f(x)=-1
% % % Position (0)
% %[-5.12,5.12]
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % function  y = func(x,DIM)
% x1 = x(1);
% x2 = x(2);
% frac1 = 1 + cos(12*sqrt(x1^2+x2^2));
% frac2 = 0.5*(x1^2+x2^2) + 2;
% y = -frac1/frac2-(-1); %把最优值平移到0
%     case 14
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Beale function
% % % The global minima: f(3,0.5)=0
% % % Position (3,0.5)
% %DIM=2
% %[-4.5,4.5]
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % function  y = func(x,DIM)
% x1 = x(1);
% x2 = x(2);
% frac1 =(1.5-x1+x1*x2)^2 ;
% frac2 = (2.25-x1+x1*x2^2)^2;
% frac3=(2.625-x1+x1*x2^3)^2;
% y = frac1+frac2+frac3;
%     case 15
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Three-hump camel  function
% % % The global minima: f(0)=0
% % % Position (0)
% %DIM=2
% %[-5,5]
% % %%%%%%%%%%%%%
% % function  y = func(x,DIM)
% x1 = x(1);
% x2 = x(2);
% y = 2*x1^2-1.05*x1^4+x1^6/6+x1*x2+x2^2;
% 
%     case 16
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % McCormick  function
% % % The global minima: f(-0.54717,-1.54719)=-1.9133
% % % Position (0)
% %DIM=2
% %[-4,4]
% % %%%%%%%%%%%%%
% % function  y = func(x,DIM)
% x1 = x(1);
% x2 = x(2);
% y = sin(x1+x2)+(x1-x2)^2-1.5*x1+2.5*x2+1-(-1.9133);  %把最优值平移到0
end


