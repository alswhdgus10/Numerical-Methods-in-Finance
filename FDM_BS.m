PutCall = 'P';
K = 10;
rf = 0.1;
D0 = 0;
sigma = 0.4;
T = 0.25;
s0 = 10;

smax = 20;
smin = 0;
M = 1600;                 % number of time point
N = 160;                  % numer of stock price point
dS = (smax-smin)/N;       % S-mesh size 
dt = T/M;                 % t-mesh size
Exactprice = BSprice(PutCall,s0,T,K,rf,D0,sigma);
fprintf('%s BSprice %2.0f %2.4f %2.4f \n',PutCall,s0,Exactprice(1),Exactprice(2));

EFDMprice = EFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N);
disp(EFDMprice);

IFDMprice = IFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N);
disp(IFDMprice);

CNprice = CNFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N);
disp(CNprice);

function price = BSprice(PutCall,s0,ttm,K,rf,D0,sigma)
d1 =(1/(sigma*sqrt(ttm)))*(log(s0/K) + (rf - D0 + 0.5*sigma^2)*ttm);
d2 = (1/(sigma*sqrt(ttm)))*(log(s0/K) + (rf - D0 - 0.5*sigma^2)*ttm);
Nd1 = normcdf(d1);
Nd2 = normcdf(d2);
Nd1m = normcdf(-d1);
Nd2m = normcdf(-d2);
    
if strcmp(PutCall,'C')
	price1 = s0*exp(-D0*ttm)*Nd1 - K*exp(-rf*ttm)*Nd2;
    price2 = Nd1;
    else if strcmp(PutCall,'P')
         price1 = K*exp(-rf*ttm)*Nd2m - s0*exp(-D0*ttm)*Nd1m;
         price2 = -Nd1m;
        end
end
price = [price1 price2];
end

% Explicit finite difference method algorithm
function price = EFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N)

v(1:M,1:N) = 0;

% BC at maturity
if strcmp(PutCall,'C')
    v(1,1:N) = max(smin+(0:N-1)*dS - K,zeros(size(1:N)));
    else if strcmp(PutCall,'P')
         v(1,1:N) = max(K - (smin+(0:N-1)*dS),zeros(size(1:N)));
    end
end

% BC at S = smax or smin
if strcmp(PutCall,'C')
    v(2:M,1) = zeros(M-1,1);                              % call option value when underlying is 0
    v(2:M,N) = (smin+(N-1)*dS)*exp(-D0*(1:M-1)*dt) - K*exp(-rf*(1:M)*dt);     % call option value when underlying is large 
else if strcmp(PutCall,'P')
     v(2:M,1) = K*exp(-rf*(2:M)*dt);                    % put option value when underlying is large
     v(2:M,N) = zeros(M-1,1);                             % put option value when underlying is 0 
    end  
end

%coefficients of matrix for EFDM
a = 1 - dt*(rf + (sigma^2)*((2:N-1).^2));
b = (dt/2)*(-(rf-D0)*(2:N-1) + (sigma^2)*((2:N-1).^2));
c = (dt/2)*((rf-D0)*(2:N-1) + (sigma^2)*((2:N-1).^2));

%Finding value at M 
for m=2:M  
    v(m,2:N-1) = b.*v(m-1,1:N-2) + a.*v(m-1,2:N-1) + c.*v(m-1,3:N);  
end

price = zeros(N,2);
price(1:N,1) = smin+(0:N-1)*dS;
price(1:N,2) = v(M,1:N);
end

% Implicit finite difference method algorithm
function price = IFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N)

v(1:M,1:N) = 0;

% BC at maturity
if strcmp(PutCall,'C')
    v(1,1:N) = max(smin+(0:N-1)*dS - K,zeros(size(1:N)));
    else if strcmp(PutCall,'P')
         v(1,1:N) = max(K - (smin+(0:N-1)*dS),zeros(size(1:N)));
    end
end

% BC at S = smax or smin
if strcmp(PutCall,'C')
    v(2:M,1) = zeros(M-1,1);                              % call option value when underlying is 0
    v(2:M,N) = (smin+(N-1)*dS)*exp(-D0*(1:M-1)*dt) - K*exp(-rf*(1:M)*dt);     % call option value when underlying is large 
else if strcmp(PutCall,'P')
     v(2:M,1) = K*exp(-rf*(2:M)*dt);                    % put option value when underlying is large
     v(2:M,N) = zeros(M-1,1);                             % put option value when underlying is 0 
    end  
end


%coefficients of matrix for IFDM
a = 1 + dt*(rf + (sigma^2)*((2:N-1).^2));
b = (dt/2)*((rf-D0)*(2:N-1) - (sigma^2)*((2:N-1).^2));
c = (dt/2)*(-(rf-D0)*(2:N-1) - (sigma^2)*((2:N-1).^2));

% a-(b+c)>0 is checked! the matrix is strictly diagonally dominant

for m=2:M 
    d = v(m-1,2:N-1);
    d(1) = d(1) - b(1)*v(m,1);
    d(N-2) = d(N-2) - c(N-2)*v(m,N);
    v(m,2:N-1) = tridiag(a,b,c,d);
end

price = zeros(N,2);
price(1:N,1) = smin+(0:N-1)*dS;
price(1:N,2) = v(M,1:N);
end

% Crank-Nicolson finite difference method algorithm
function price = CNFDM(PutCall,smin,K,rf,D0,sigma,dt,dS,M,N)

v(1:M,1:N) = 0;

% BC at maturity
if strcmp(PutCall,'C')
    v(1,1:N) = max(smin+(0:N-1)*dS - K,zeros(size(1:N)));
    else if strcmp(PutCall,'P')
         v(1,1:N) = max(K - (smin+(0:N-1)*dS),zeros(size(1:N)));
    end
end

% BC at S = smax or smin
if strcmp(PutCall,'C')
    v(2:M,1) = zeros(M-1,1);                              % call option value when underlying is 0
    v(2:M,N) = (smin+(N-1)*dS)*exp(-D0*(1:M-1)*dt) - K*exp(-rf*(1:M)*dt);     % call option value when underlying is large 
else if strcmp(PutCall,'P')
     v(2:M,1) = K*exp(-rf*(2:M)*dt);                    % put option value when underlying is large
     v(2:M,N) = zeros(M-1,1);                           % put option value when underlying is 0 
    end  
end

%coefficients of matrix for IFDM
a = (dt/2)*(rf + (sigma^2)*((2:N-1).^2));
b = (dt/4)*((rf-D0)*(2:N-1) - (sigma^2)*((2:N-1).^2));
c = (dt/4)*(-(rf-D0)*(2:N-1) - (sigma^2)*((2:N-1).^2));

% a-(b+c)>0, the matrix is strictly diagonally dominant

for m=2:M 
    d = b.*v(m-1,1:N-2) + (-1 + a).*v(m-1,2:N-1) + c.*v(m-1,3:N);
    v(m,2:N-1) = tridiag((-1-a),-b,-c,d);
end

price = zeros(N,2);
price(1:N,1) = smin+(0:N-1)*dS;
price(1:N,2) = v(M,1:N);
end

