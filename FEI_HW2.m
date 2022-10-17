%==========================================================================
% Financial Econometric I
% PhD in Finance
% Homework 2 QMLE
%
% Maxime Borel, Florian Perusset & Keyu Zang
% Date: October 2022
%==========================================================================
clear; 
clc; 
close all;

restoredefaultpath      % restore search path to default
addpath(genpath(pwd+"/Kevin Sheppard Toolbox"))
clear RESTOREDEFAULTPATH_EXECUTED

%setting some string variable for table purpose
options = optimset('Display','Off','Diagnostics','off');
varnamesab={'10','20','30','40','50','60','70','80','90','100'};
obsnamesab={'ML','QML'};
varnamesre={'muRE','VarRE','10','20','30','40','50','60','70','80','90','100'};
obsnamesre={'{\alpha}','{\beta}'};

%% 2.a. Relative Efficiency under Normality
T=5000; %number of simulation
alpha=0.1; %parameter of the simulated process
beta=0.8;
w=1-alpha-beta;
unsigma=w/(1-alpha-beta); %by definition is equal to 1
sigma02=1;
r0=0;

zt.normal=randn(T,T); %innovation process with normal
rt.normal=zeros(T,T); %vector to store the return, initializing vector
sigmat2.normal=zeros(T,T);
garch.normal.ml.param=zeros(T,2);
garch.normal.ml.vcv=zeros(T,2);
garch.normal.qml.rvcv=zeros(T,2);

for j=1:T
    sigmat2.normal(1,j)=w + alpha*r0^2 + beta*sigma02; %compute the first value for each simulation sigma and rt
    rt.normal(1,j)=sqrt(sigmat2.normal(1,j))*zt.normal(1,j);
    
    for i=2:T %do the rest recursivly 
        sigmat2.normal(i,j)=w + alpha*rt.normal(i-1,j)^2 + beta*sigmat2.normal(i-1,j);
        rt.normal(i,j)=sqrt(sigmat2.normal(i,j))*zt.normal(i,j);
    end
    
[param,~,~,VCVrobust,VCV]=tarch(rt.normal(:,j),1,0,1,'NORMAL',2, [],options); %estime the GARCH parameter with normal
    garch.normal.ml.param(j,1)=param(2); %store the parameter
    garch.normal.ml.param(j,2)=param(3);
    
    garch.normal.ml.vcv(j,1)=VCV(2,2); %store the diagonal standard variance of the parameter 
    garch.normal.ml.vcv(j,2)=VCV(3,3);    

    garch.normal.qml.rvcv(j,1)=VCVrobust(2,2); %store the robust variance for parameter on each diagonal
    garch.normal.qml.rvcv(j,2)=VCVrobust(3,3);   
end


f1=figure(); %historgram of the distribution for variance each parameter for both method ml and qml
subplot(2,2,1)
histogram(garch.normal.ml.vcv(:,1))
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
hold on
subplot(2,2,2)
histogram(garch.normal.qml.rvcv(:,1))
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
hold on
subplot(2,2,3)
histogram(garch.normal.ml.vcv(:,2))
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
hold on
subplot(2,2,4)
histogram(garch.normal.qml.rvcv(:,2))
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
hold off
saveas(f1,'results/histnormalsigmapara','png');

f2=figure(); %same as above but for the distribution using ksdensity
subplot(2,2,1)
ksdensity(garch.normal.ml.vcv(:,1));
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
subplot(2,2,2)
ksdensity(garch.normal.qml.rvcv(:,1));
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
subplot(2,2,3)
ksdensity(garch.normal.ml.vcv(:,2));
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
subplot(2,2,4)
ksdensity(garch.normal.qml.rvcv(:,2));
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
saveas(f2,'results/distnormalsigmapara','png');

re.norm.alpha=garch.normal.ml.vcv(:,1)  ./ garch.normal.qml.rvcv(:,1); %compute the relative efficiency alpha and beta
re.norm.beta= garch.normal.ml.vcv(:,2) ./ garch.normal.qml.rvcv(:,2);

avg.re.norm.alpha=mean(re.norm.alpha); %compute the average of the relative efficiency and the variance 
avg.re.norm.beta=mean(re.norm.beta);
sig.re.norm.alpha=std(re.norm.alpha)^2;
sig.re.norm.beta=std(re.norm.beta)^2;
re.norm.sumstat=[avg.re.norm.alpha,sig.re.norm.alpha;avg.re.norm.beta,sig.re.norm.beta]; % create a table to store it 

f3=figure(); %distribution for the relative efficiency 
subplot(1,2,1)
ksdensity(re.norm.alpha);
title('Distribution of RE_{\alpha}','FontWeight','normal');
subplot(1,2,2)
ksdensity(re.norm.beta);
title('Distribution of RE_{\beta}','FontWeight','normal');
saveas(f3,'results/distnormalre','png');

n=10; %number of quantile desire
quant.normal.ml.alpha=zeros(1,n); %initialite vector to store all quantile
quant.normal.ml.beta=zeros(1,n);
quant.normal.qml.alpha=zeros(1,n);
quant.normal.qml.beta=zeros(1,n);
quant.normal.RE.alpha=zeros(1,n);
quant.normal.RE.beta=zeros(1,n);

for i=1:n %compute all the quantile for qml and ml and alpha and beta
    quant.normal.ml.alpha(1,i)=quantile(garch.normal.ml.vcv(:,1),i./n);
    quant.normal.ml.beta(1,i)=quantile(garch.normal.ml.vcv(:,2),i./n);
    quant.normal.qml.alpha(1,i)=quantile(garch.normal.qml.rvcv(:,1),i./n);
    quant.normal.qml.beta(1,i)=quantile(garch.normal.qml.rvcv(:,2),i./n);
    quant.normal.RE.alpha(1,i)=quantile(re.norm.alpha,i./n);
    quant.normal.RE.beta(1,i)=quantile(re.norm.beta,i./n);
end

result.normal.re=[re.norm.sumstat,[quant.normal.RE.alpha;quant.normal.RE.beta]]; % create table to export result 
result.normal.re=mat2dataset(result.normal.re,'VarNames',varnamesre,'ObsNames',obsnamesre);
export(result.normal.re,'file','results/resultnormalre.xls')
result.normal.alpha=[quant.normal.ml.alpha;quant.normal.qml.alpha];
result.normal.alpha=mat2dataset(result.normal.alpha,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.normal.alpha,'file','results/resultnormalalpha.xls')
result.normal.beta=[quant.normal.ml.beta;quant.normal.qml.beta];
result.normal.beta=mat2dataset(result.normal.beta,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.normal.beta,'file','results/resultnormalbeta.xls')
%% 2.b. Relative Efficiency under t distribution
%the code below is essentially a copy on the above juste change the
%distribution and the degree of freedom for nu small change arise as there
%is one additional parameter. 
%% nu=5
zt.t5=trnd(5,[T,T]);
rt.t5=zeros(T,T);
sigmat2.t5=zeros(T,T);

garch.t5.ml.param=zeros(T,2);
garch.t5.ml.vcv=zeros(T,2);
garch.t5.qml.param=zeros(T,2);
garch.t5.qml.rvcv=zeros(T,2);

for j=1:T
    sigmat2.t5(1,j)=w + alpha*r0^2 + beta*sigma02;
    rt.t5(1,j)=sqrt(sigmat2.t5(1,j))*zt.t5(1,j);
    
    for i=2:T
        sigmat2.t5(i,j)=w + alpha*rt.t5(i-1,j)^2 + beta*sigmat2.t5(i-1,j);
        rt.t5(i,j)=sqrt(sigmat2.t5(i,j))*zt.t5(i,j);
    end
    
[para.qml,~,~,VCVrobust]=tarch(rt.t5(:,j),1,0,1,'NORMAL',2, [],options);
    garch.t5.qml.param(j,1)=para.qml(2);
    garch.t5.qml.param(j,2)=para.qml(3);

    garch.t5.qml.rvcv(j,:)=[VCVrobust(2,2) VCVrobust(3,3)];
    
[para.ml,~,~,~,VCV]=tarch(rt.t5(:,j),1,0,1,'STUDENTST',2, [],options);
    garch.t5.ml.param(j,1)=para.ml(2);
    garch.t5.ml.param(j,2)=para.ml(3);

    garch.t5.ml.vcv(j,:)=[VCV(2,2) VCV(3,3)];
end


f4=figure();
subplot(2,2,1)
histogram(garch.t5.ml.vcv(:,1))
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
hold on
subplot(2,2,2)
histogram(garch.t5.qml.rvcv(:,1))
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
hold on
subplot(2,2,3)
histogram(garch.t5.ml.vcv(:,2))
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
hold on
subplot(2,2,4)
histogram(garch.t5.qml.rvcv(:,2))
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
hold off
saveas(f4,'results/histt5sigmapara','png');

f5=figure();
subplot(2,2,1)
ksdensity(garch.t5.ml.vcv(:,1));
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
subplot(2,2,2)
ksdensity(garch.t5.qml.rvcv(:,1));
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
subplot(2,2,3)
ksdensity(garch.t5.ml.vcv(:,2));
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
subplot(2,2,4)
ksdensity(garch.t5.qml.rvcv(:,2));
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
saveas(f5,'results/distt5sigmapara','png');

re.t5.alpha=garch.t5.ml.vcv(:,1)./garch.t5.qml.rvcv(:,1);
re.t5.beta=garch.t5.ml.vcv(:,2)./garch.t5.qml.rvcv(:,2);


avg.re.t5.alpha=mean(re.t5.alpha);
avg.re.t5.beta=mean(re.t5.beta);
sig.re.t5.alpha=std(re.t5.alpha)^2;
sig.re.t5.beta=std(re.t5.beta)^2;
re.t5.sumstat=[avg.re.t5.alpha, sig.re.t5.alpha;avg.re.t5.beta, sig.re.t5.beta];

f6=figure();
subplot(1,2,1)
ksdensity(re.t5.alpha);
title('Distribution of RE_{\alpha}','FontWeight','normal');
subplot(1,2,2)
ksdensity(re.t5.beta);
title('Distribution of RE_{\beta}','FontWeight','normal');
saveas(f6,'results/distt5re','png');

n=10;
quant.t5.ml.alpha=zeros(1,n);
quant.t5.ml.beta=zeros(1,n);
quant.t5.qml.alpha=zeros(1,n);
quant.t5.qml.beta=zeros(1,n);
quant.t5.RE.alpha=zeros(1,n);
quant.t5.RE.beta=zeros(1,n);

for i=1:n
    quant.t5.ml.alpha(1,i)=quantile(garch.t5.ml.vcv(:,1),i./n);
    quant.t5.ml.beta(1,i)=quantile(garch.t5.ml.vcv(:,2),i./n);
    quant.t5.qml.alpha(1,i)=quantile(garch.t5.qml.rvcv(:,1),i./n);
    quant.t5.qml.beta(1,i)=quantile(garch.t5.qml.rvcv(:,2),i./n);
    quant.t5.RE.alpha(1,i)=quantile(re.t5.alpha,i./n);
    quant.t5.RE.beta(1,i)=quantile(re.t5.beta,i./n);
end

result.t5.re=[re.t5.sumstat,[quant.t5.RE.alpha;quant.t5.RE.beta]];
result.t5.re=mat2dataset(result.t5.re,'VarNames',varnamesre,'ObsNames',obsnamesre);
export(result.t5.re,'file','results/resultt5re.xls')
result.t5.alpha=[quant.t5.ml.alpha;quant.t5.qml.alpha];
result.t5.alpha=mat2dataset(result.t5.alpha,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t5.alpha,'file','results/resultt5alpha.xls')
result.t5.beta=[quant.t5.ml.beta;quant.t5.qml.beta];
result.t5.beta=mat2dataset(result.t5.beta,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t5.beta,'file','results/result5beta.xls')
%% nu=8
zt.t8=trnd(8,[T,T]);
rt.t8=zeros(T,T);
sigmat2.t8=zeros(T,T);

garch.t8.ml.param=zeros(T,2);
garch.t8.ml.vcv=zeros(T,2);
garch.t8.qml.param=zeros(T,2);
garch.t8.qml.rvcv=zeros(T,2);

for j=1:T
    sigmat2.t8(1,j)=w + alpha*r0^2 + beta*sigma02;
    rt.t8(1,j)=sqrt(sigmat2.t8(1,j))*zt.t8(1,j);
    
    for i=2:T
        sigmat2.t8(i,j)=w + alpha*rt.t8(i-1,j)^2 + beta*sigmat2.t8(i-1,j);
        rt.t8(i,j)=sqrt(sigmat2.t8(i,j))*zt.t8(i,j);
    end
    
[para.qml,~,~,VCVrobust]=tarch(rt.t8(:,j),1,0,1,'NORMAL',2, [],options);
    garch.t8.qml.param(j,:)=para.qml(2:3)';
    garch.t8.qml.rvcv(j,:)=diag(VCVrobust(2:3,2:3))';
    
[para.ml,~,~,~,VCV]=tarch(rt.t8(:,j),1,0,1,'STUDENTST',2, [],options);
    garch.t8.ml.param(j,:)=para.ml(2:3)';
    garch.t8.ml.vcv(j,:)=diag(VCV(2:3,2:3))';
end


f7=figure();
subplot(2,2,1)
histogram(garch.t8.ml.vcv(:,1))
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
hold on
subplot(2,2,2)
histogram(garch.t8.qml.rvcv(:,1))
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
hold on
subplot(2,2,3)
histogram(garch.t8.ml.vcv(:,2))
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
hold on
subplot(2,2,4)
histogram(garch.t8.qml.rvcv(:,2))
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
hold off
saveas(f7,'results/histt8sigmapara','png');

f8=figure();
subplot(2,2,1)
ksdensity(garch.t8.ml.vcv(:,1));
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
subplot(2,2,2)
ksdensity(garch.t8.qml.rvcv(:,1));
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
subplot(2,2,3)
ksdensity(garch.t8.ml.vcv(:,2));
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
subplot(2,2,4)
ksdensity(garch.t8.qml.rvcv(:,2));
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
saveas(f8,'results/distt8sigmapara','png');

re.t8.alpha=garch.t8.ml.vcv(:,1)./garch.t8.qml.rvcv(:,1);
re.t8.beta=garch.t8.ml.vcv(:,2)./garch.t8.qml.rvcv(:,2);

avg.re.t8.alpha=mean(re.t8.alpha);
avg.re.t8.beta=mean(re.t8.beta);
sig.re.t8.alpha=std(re.t8.alpha)^2;
sig.re.t8.beta=std(re.t8.beta)^2;
re.t8.sumstat=[avg.re.t8.alpha, sig.re.t8.alpha;avg.re.t8.beta, sig.re.t8.beta];

f9=figure();
subplot(1,2,1)
ksdensity(re.t8.alpha);
title('Distribution of RE_{\alpha}','FontWeight','normal');
subplot(1,2,2)
ksdensity(re.t8.beta);
title('Distribution of RE_{\beta}','FontWeight','normal');
saveas(f9,'results/distt8re','png');

n=10;
quant.t8.ml.alpha=zeros(1,n);
quant.t8.ml.beta=zeros(1,n);
quant.t8.qml.alpha=zeros(1,n);
quant.t8.qml.beta=zeros(1,n);
quant.t8.RE.alpha=zeros(1,n);
quant.t8.RE.beta=zeros(1,n);

for i=1:n
    quant.t8.ml.alpha(1,i)=quantile(garch.t8.ml.vcv(:,1),i./n);
    quant.t8.ml.beta(1,i)=quantile(garch.t8.ml.vcv(:,2),i./n);
    quant.t8.qml.alpha(1,i)=quantile(garch.t8.qml.rvcv(:,1),i./n);
    quant.t8.qml.beta(1,i)=quantile(garch.t8.qml.rvcv(:,2),i./n);
    quant.t8.RE.alpha(1,i)=quantile(re.t8.alpha,i./n);
    quant.t8.RE.beta(1,i)=quantile(re.t8.beta,i./n);
end

result.t8.re=[re.t8.sumstat,[quant.t8.RE.alpha;quant.t8.RE.beta]];
result.t8.re=mat2dataset(result.t8.re,'VarNames',varnamesre,'ObsNames',obsnamesre);
export(result.t8.re,'file','results/resultt8re.xls')
result.t8.alpha=[quant.t8.ml.alpha;quant.t8.qml.alpha];
result.t8.alpha=mat2dataset(result.t8.alpha,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t8.alpha,'file','results/resultt8alpha.xls')
result.t8.beta=[quant.t8.ml.beta;quant.t8.qml.beta];
result.t8.beta=mat2dataset(result.t8.beta,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t8.beta,'file','results/result8beta.xls')
%% nu=12
zt.t12=trnd(12,[T,T]);
rt.t12=zeros(T,T);
sigmat2.t12=zeros(T,T);

garch.t12.ml.param=zeros(T,2);
garch.t12.ml.vcv=zeros(T,2);
garch.t12.qml.param=zeros(T,2);
garch.t12.qml.rvcv=zeros(T,2);

for j=1:T
    sigmat2.t12(1,j)=w + alpha*r0^2 + beta*sigma02;
    rt.t12(1,j)=sqrt(sigmat2.t12(1,j))*zt.t12(1,j);
    
    for i=2:T
        sigmat2.t12(i,j)=w + alpha*rt.t12(i-1,j)^2 + beta*sigmat2.t12(i-1,j);
        rt.t12(i,j)=sqrt(sigmat2.t12(i,j))*zt.t12(i,j);
    end
    
[para.qml,~,~,VCVrobust]=tarch(rt.t12(:,j),1,0,1,'NORMAL',2, [],options);
    garch.t12.qml.param(j,:)=para.qml(2:3)';
    garch.t12.qml.rvcv(j,:)=diag(VCVrobust(2:3,2:3))';
    
[para.ml,~,~,~,VCV]=tarch(rt.t12(:,j),1,0,1,'STUDENTST',2, [],options);
    garch.t12.ml.param(j,:)=para.ml(2:3)';
    garch.t12.ml.vcv(j,:)=diag(VCV(2:3,2:3))';
end


f10=figure();
subplot(2,2,1)
histogram(garch.t12.ml.vcv(:,1))
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
hold on
subplot(2,2,2)
histogram(garch.t12.qml.rvcv(:,1))
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
hold on
subplot(2,2,3)
histogram(garch.t12.ml.vcv(:,2))
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
hold on
subplot(2,2,4)
histogram(garch.t12.qml.rvcv(:,2))
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
hold off
saveas(f10,'results/histt12sigmapara','png');

f11=figure();
subplot(2,2,1)
ksdensity(garch.t12.ml.vcv(:,1));
title('Distribution of \sigma^2_{\alpha,ML}','FontWeight','normal');
subplot(2,2,2)
ksdensity(garch.t12.qml.rvcv(:,1));
title('Distribution of \sigma^2_{\alpha,QML}','FontWeight','normal');
subplot(2,2,3)
ksdensity(garch.t12.ml.vcv(:,2));
title('Distribution of \sigma^2_{\beta,ML}','FontWeight','normal');
subplot(2,2,4)
ksdensity(garch.t12.qml.rvcv(:,2));
title('Distribution of \sigma^2_{\beta,QML}','FontWeight','normal');  
saveas(f11,'results/distt12sigmapara','png');

re.t12.alpha=garch.t12.ml.vcv(:,1)./garch.t12.qml.rvcv(:,1);
re.t12.beta=garch.t12.ml.vcv(:,2)./garch.t12.qml.rvcv(:,2);

avg.re.t12.alpha=mean(re.t12.alpha);
avg.re.t12.beta=mean(re.t12.beta);
sig.re.t12.alpha=std(re.t12.alpha)^2;
sig.re.t12.beta=std(re.t12.beta)^2;
re.t12.sumstat=[avg.re.t12.alpha, sig.re.t12.alpha;avg.re.t12.beta, sig.re.t12.beta];

f12=figure();
subplot(1,2,1)
ksdensity(re.t12.alpha);
title('Distribution of RE_{\alpha}','FontWeight','normal');
subplot(1,2,2)
ksdensity(re.t12.beta);
title('Distribution of RE_{\beta}','FontWeight','normal');
saveas(f12,'results/distt12re','png');

n=10;
quant.t12.ml.alpha=zeros(1,n);
quant.t12.ml.beta=zeros(1,n);
quant.t12.qml.alpha=zeros(1,n);
quant.t12.qml.beta=zeros(1,n);
quant.t12.RE.alpha=zeros(1,n);
quant.t12.RE.beta=zeros(1,n);

for i=1:n
    quant.t12.ml.alpha(1,i)=quantile(garch.t12.ml.vcv(:,1),i./n);
    quant.t12.ml.beta(1,i)=quantile(garch.t12.ml.vcv(:,2),i./n);
    quant.t12.qml.alpha(1,i)=quantile(garch.t12.qml.rvcv(:,1),i./n);
    quant.t12.qml.beta(1,i)=quantile(garch.t12.qml.rvcv(:,2),i./n);
    quant.t12.RE.alpha(1,i)=quantile(re.t12.alpha,i./n);
    quant.t12.RE.beta(1,i)=quantile(re.t12.beta,i./n);
end

result.t12.re=[re.t12.sumstat,[quant.t12.RE.alpha;quant.t12.RE.beta]];
result.t12.re=mat2dataset(result.t12.re,'VarNames',varnamesre,'ObsNames',obsnamesre);
export(result.t12.re,'file','results/resultt12re.xls')
result.t12.alpha=[quant.t12.ml.alpha;quant.t12.qml.alpha];
result.t12.alpha=mat2dataset(result.t12.alpha,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t12.alpha,'file','results/resultt12alpha.xls')
result.t12.beta=[quant.t12.ml.beta;quant.t12.qml.beta];
result.t12.beta=mat2dataset(result.t12.beta,'VarNames',varnamesab,'ObsNames',obsnamesab);
export(result.t12.beta,'file','results/result12beta.xls')