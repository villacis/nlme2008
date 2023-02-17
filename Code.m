%% Matlab code for estimating Linear Response Stochastic Plateau (LRSP) functions as presented in: 
%  Tembo, G., Brorsen, B. W., Epplin, F. M., & Tostão, E. (2008). 
%  Crop input response functions with stochastic plateaus. 
%  American Journal of Agricultural Economics, 90(2), 424-434.

%  Code Created by Alexis Villacis
%  Email: alexis.villacis@asu.edu 

%% If you benefit from this code, make sure to cite this paper that motivated me to write and share this code:
%  Villacis, A. H., Ramsey, A. F., Delgado, J. A., & Alwang, J. R. (2020). 
%  Estimating Economically Optimal Levels of Nitrogen Fertilizer in No-Tillage Continuous Corn. 
%  Journal of Agricultural and Applied Economics, 52(4), 613-623.

%  Feel free to reach out to me at: alexis.villacis@asu.edu if you have any questions or find any bugs.

%%Thank you!!!

%%
%%%%%%%%
% DATA %
%%%%%%%%

% Make sure to download the data file "data.mat" available in github. 
% That is the data file you need in order to replicate the example I discuss in my website:  
% https://alexisvillacis.wordpress.com/2020/09/07/an-application-of-non-linear-mixed-effects-models-using-matlab/

%Load & prepare data
load /Users/ahvillacis/Documents/MATLAB/data.mat; %change this pathname 

%Variable List

%1  Year               
%2  N Rate kg/ha                    
%3  Grain Yield kg/ha

year=data(:,1);
nit=data(:,2);
yield=data(:,3);

% Plot the scatter plot of yield  vs nitrogen grouped by year practice.
figure(1);
gscatter(nit,yield,year)
axis([0 280 0 14000])
xlabel('Nitrogen (kg/ha)')
ylabel('Grain Yield (kg/ha)')
legend('Location','northeast outside')
ax = gca
ax.TitleFontSizeMultiplier = 1.2
title('Grain Yield vs Nitrogen by Year')

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MODEL USING LINEAR + PLATEAU MODEL W/O Random Effects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct the model via an anonymous function.
NLmodel=@(betaNL,nit) min((betaNL(1)+betaNL(2)*nit), (betaNL(3))); 

% Use the |fitnlm| function to fit the model to all of the data, ignoring
% year-specific effects.
betaNLi = [4000 20 10000];
NL1 = fitnlm(nit,yield,NLmodel,betaNLi);

%Display summary (including mean squared error) & save betas and residuals
NL1;
betaNL1 = NL1.Coefficients.Estimate;
resNL1 = NL1.Residuals.Raw;
npNL1=(betaNL1(3)-betaNL1(1))/(betaNL1(2));

%%
% Super impose the model on the scatter plot of data.
figure(2);
scatter(nit,yield,30,'b')
axis([0 280 0 14000])
hold on
tplot = 0:0.01:280;
plot(tplot,NLmodel(betaNL1,tplot),'k','LineWidth',3)
xlabel('Nitrogen (kg/ha)')
ylabel('Grain Yield (kg/ha)')
ax = gca
hold off

%%
% Draw the box-plot of residuals by year.
figure(3);
h = boxplot(resNL1,year,'symbol','o');
set(h(~isnan(h)),'LineWidth',2)
hold on
boxplot(resNL1,year,'colors','k','symbol','ko')
grid on
xlabel('Year')
ylabel('Residual')

% The box plot of residuals by year shows that the boxes are mostly
% above or below zero, indicating that the model has failed to account for
% year-specific effects.


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MODEL USING LINEAR + PLATEAU MODEL WITH Random Effects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Use the |nlmefit| function to fit a non-linear mixed-effects model to 
%the data. You can also use |nlmefitsa| in place of |nlmefit| .

% The following anonymous function, |NLMEmodel| , adapts the
% three-parameter model used by |fitnlm| to the calling syntax of |nlmefit|
% by allowing separate parameters for each YEAR. By default,
% |nlmefit| assigns random effects to all the model parameters. 

% Construct the nlme model via an anonymous function.
NLMEmodel = @(betaNLME,nit) min((betaNLME(1)+betaNLME(2)*nit), (betaNLME(3)));   
                                    
%% Model NLME1
% By default, |nlmefit| assumes a diagonal covariance matrix (no covariance
% among the random effects) to avoid overparametrization and related
% convergence issues.

[betaNLME1,PSI1,stats1] = nlmefit(nit,yield,year, ...
                          [],NLMEmodel,betaNL1)

% The estimated covariance matrix |PSI| shows that the variance of the
% all random effects are different from zero, suggesting that you can't 
% remove any parameter to simplify the model. 

%% Model NLME2
% We use REParamsSelect to specify Random effects for only the 
% first parameter (intercept) and third parameter (plateau)

[betaNLME2,PSI2,stats2] = nlmefit(nit,yield,year, ...
                          [],NLMEmodel,betaNLME1, ...
                         'REParamsSelect',[1 3])

%% Model NLME3
% Refitting the simplified model with a full covariance matrix allows for
% identification of correlations among the random effects. To do this, use
% the |CovPattern| parameter to specify the pattern of nonzero elements in
% the covariance matrix.

[betaNLME3,PSI3,stats3] = nlmefit(nit,yield,year, ...
                          [],NLMEmodel,betaNLME1, ...
                          'CovPattern',ones(3))
                     
%% Model NLME4
% The default method to approximate the likelihood of the model is
% LME. LME uses the likelihood for the linear mixed-effects model at the 
% current conditional estimates of beta and B. NLMEFIT also provides 
% the option to use FOCE – First order Laplacian approximation at the 
%conditional estimates of B, which is equivalent to Gaussian quadrature.

[betaNLME4,PSI4,stats4] = nlmefit(nit,yield,year, ...
                          [],NLMEmodel,betaNLME3, ...
                          'CovPattern',ones(3), ... 
                          'ApproximationType','FOCE')

%Save residulas
resNLME4=stats4.ires
%%
% Draw the box-plot of residuals by year.
figure(4);
h = boxplot(resNLME4,year,'symbol','o');
set(h(~isnan(h)),'LineWidth',2)
hold on
boxplot(resNLME4,year,'colors','k','symbol','ko')
grid on
xlabel('Year')
ylabel('Residual')

% The box plot of residuals by year shows that the boxes are centered at zero, 
% indicating that the model has accounted for year-specific effects.  

% The structure in the covariance matrix is more apparent if you convert |PSI|
% to a correlation matrix using |corrcov| .
RHO4 = corrcov(PSI4)

% Plot a histogram of the individual errors for a visual assessment of the pdf

figure(5);
histogram(stats4.ires)
title('NLME4: Histogram of Individual Errors')

% Create a normal probability plot of the individual errors.

figure(6);
normplot(stats4.ires)
title('NLME4: Normal Probability Plot of Residuals')

% Calculate Skewness and Kurtosis

skeNLME4 = skewness(stats4.ires)
kurtNLME4 = kurtosis(stats4.ires)
