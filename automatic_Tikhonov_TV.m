function [m, out] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options)
%
% [m, out] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options)
% [m, out] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit)
%
% Computes the solution to the problems 
% 
% min f(m) subject to || G m - d ||^2 = epsilon,
%
% where
%
% [TT] : f(m) = ||D1 m1||_1 + beta/2||D2 m2||_2^2
% [TV] : f(m) = ||D1 m1||_1
% [T]  : f(m) = beta/2||D2 m2||_2^2
%
% and where D1 and D2 are scaled finite difference discretizations of the 
% gradient and partial second derivative operators, and beta is a balancing
% parameter that can be estimated adaptively.
%
% Inputs:
%  G : a full or sparse matrix
%  d : right-hand side vector
%  epsilon : (estimate of) the squared 2-norm of the noise
%  mu : vector of penalty parameter for the Lagrangian
%  maxit : maximum number of iterations for ADMM
%  options : (optional) structure with the following fields 
%      fullmatrix - G should be treated as full or sparse matrix
%                   [ {'on'} | 'off' ]
%      m_true     - true solution; allows us to returns error norms with
%                   respect to x_true at each iteration
%                   [ array | {0} ]
%      beta0      - first value for adaptive estimation for beta
%                   [ {1} | nonnegative scalar ]
%      NoStop     - specifies whether the iterations should proceed after
%                   a stopping criterion has been satisfied
%                   [ 'on' | {'off'} ]
%      StopThr    - threshold for a stopping criterion based on the
%                   stabilization of the solution
%                   [ {1e-4} | nonnegative scalar ]
%      type       - optimization problem to be solved
%                   [ {'TT'} | 'TV' | 'T' ]
%      plotty     - wether the progress of the solver should be plotted at
%                   each iteration
%                   [ 'on' | {'off'} ]
%      beta       - adaptive estimation or fixed value for beta
%                   [ nonnegative scalar | {'adapt'} ]
%
% Outputs:
%   m : computed solution
%   out : structure with the following fields:
%      stopit   - iteration satisfying the stopping criterion
%      saved_iterations - iteration numbers of iterates stored in X 
%      diffm    - array of relative 2-norm differences between two 
%                 consecutive solutions 
%      Enrm     - relative error norms (requires m_true) at each iteration
%      Rnrm2    - array of 2-norm squared residuals
%      phi      - array of values of the function phi at each iteration
%      beta     - array of computed values of beta
%
% Silvia Gazzola, University of Bath
% Ali Gholami, University of Teheran
% April, 2022

if nargin == 5
    % set default input options
    options.fullmatrix = 'on';
    options.m_true = 0;
    options.beta0 = 1;
    options.NoStop = 'on';
    options.StopThr = 1e-4;
    options.type = 'TT';
    options.plotty = 'off';
    options.beta = 'adapt';
    options.zthr = 2.5;
end

adaptbeta = strcmp(options.beta, 'adapt');


mu1 = mu(1); mu2 = mu(2); mu3 = mu(3);
m_true = options.m_true;

n = size(G,2);
sqrtn = sqrt(n);
if sqrtn ~= floor(sqrtn)
    error('This cide can only work with square 2D quantities')
end
[D1, D1_bar] = discr_grad(sqrtn);


if strcmp(options.fullmatrix, 'on')
    B = mu1*(D1'*(D1)) + mu2*(G'*(G)); % eq. (2.11)
end

% initial guess for Lagrange multipliers (dual variables)
lambda_1 = zeros(2*n,1);
lambda_2 = zeros(numel(d),1); %
lambda_3 = 0;

% initial guess for primal variables
m = zeros(n,1);
g1 = zeros(2*n,1);
g2 = zeros(2*n,1);
e = zeros(numel(d),1);

% storing variables
if max(abs(m_true)) ~=0, Enrm = zeros(maxit, 1); end
Rnrm2 = zeros(maxit, 1);
beta = zeros(maxit+1, 1);
phi = zeros(maxit, 1);
diffm = zeros(maxit, 1);

if ~strcmp(options.type, 'TV'), beta(1) = options.beta0; end
stopit = 0;
if strcmp(options.plotty, 'on'), figure; end

% running iterations
for k=1:maxit
    disp(k)
    %----------------------------m-subproblem ----------------------------
    if strcmp(options.fullmatrix, 'on')
        rhs = mu1*D1'*(g1 + g2 + lambda_1) + mu2*G'*(d(:) + e + lambda_2) ;
        m = B\rhs;
    else
        B = [sqrt(mu1)*D1; sqrt(mu2)*G];
        rhs = [sqrt(mu1)*(g1 + g2 + lambda_1); sqrt(mu2)*(d(:) + e + lambda_2)];
        optcgls = IRset('x0', m, 'MaxIter', 200, 'NE_Rtol', 1e-7,...
            'IterBar', 'off', 'verbosity', 'off');
        m = IRcgls(B, rhs, optcgls);
    end
    if max(abs(m_true))~=0
        Enrm(k) = norm(m - m_true(:))/norm(m_true(:));
    end
    if k == 1, m_prev = m; end
    if k>1 
        diffm(k) = norm(m - m_prev)/norm(m_prev); 
        m_prev = m;
        if diffm(k)<options.StopThr && stopit == 0
            stopit = k-1;
            if strcmp(options.NoStop, 'off')
                m = m_prev;
                if max(abs(m_true))~=0, Enrm = Enrm(1:stopit); end
                Rnrm2 = Rnrm2(1:stopit);
                phi = phi(1:stopit);
                beta = beta(1:stopit);
                diffm = diffm(1:stopit);
                break
            end
        end
    end
    %----------------------------g1-subproblem ----------------------------
    if strcmp(options.type, 'TT') || strcmp(options.type, 'TV')
        y1 = D1*m  - g2 - lambda_1;
        g1 = sign(y1).*max(abs(y1)-1/mu1,0);
    end
    %----------------------------g2-subproblem ----------------------------
    if strcmp(options.type, 'TT') || strcmp(options.type, 'T')
        y2 = D1*m  - g1 - lambda_1;
        g2 =  (speye(2*n) + (beta(k)/mu1)*(D1_bar'*D1_bar))\y2;
    end
    %-----------------------------e-subproblem ----------------------------
    y = (G*m - d(:) - lambda_2);
    E = norm(y)^2;
    pp = (mu2 - 2*mu3*(epsilon + lambda_3))/(2*mu3*E);
    qq = -mu2/(2*mu3*E);
    kappa = roots([1 0 pp qq]);
    gamma = max(kappa(kappa == real(kappa)));
    e = gamma*y;
    Rnrm2(k) = norm(G*m - d(:))^2;
    %-----------------------------dual-subproblem -------------------------
    lambda_1 = lambda_1 + g1 + g2 - D1*m;
    lambda_2 = lambda_2 + d(:) + e - G*m;
    lambda_3 = lambda_3 + epsilon - norm(e)^2;
    %----------------------- beta-update ----------------------------------
    if adaptbeta
        if strcmp(options.type, 'TT')
            g = D1*m;
            target = zscore(g, options.zthr); % ||normal(g)||_inf
            value = norm(g2,'inf');
            phi(k) = value - target;
            beta(k+1)= 2*value/(value + target)*beta(k);
        elseif strcmp(options.type, 'T')
            beta(k+1) = 1;
        end
    else
        beta(k+1) = options.beta;
    end
    %------------------------------------------------------------------
    if strcmp(options.plotty, 'on')
    % plotting the progress live
        if strcmp(options.type, 'TT')
        subplot(231)
        mesh(reshape(m_true,sqrtn,sqrtn))
        title('True')
        subplot(232)
        mesh(reshape(m,sqrtn,sqrtn));% axis([170 210 170 230])
        title('Estimate')
        subplot(233)
        plot(1:k,Rnrm2(1:k),'b');hold on; plot([0 k],[epsilon epsilon],'--r')%;
        ylabel('$||e||_2^2$','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        hold off
        subplot(234)
        semilogx(1:k,phi(1:k),'linewidth',1)
        ylabel('$\phi(k)$','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        drawnow
        subplot(235)
        plot(1:k,beta(1:k),'linewidth',1)
        ylabel('$\beta(k)$','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        subplot(236)
        plot(1:k,Enrm(1:k),'linewidth',1)
        ylabel('Model error','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        drawnow
        elseif strcmp(options.type, 'TV') || strcmp(options.type, 'T')
        subplot(221)
        mesh(reshape(m_true,sqrtn,sqrtn))
        title('True')
        subplot(222)
        mesh(reshape(m,sqrtn,sqrtn));% axis([170 210 170 230])
        title('Estimate')
        subplot(223)
        plot(1:k,Rnrm2(1:k),'b');hold on; plot([0 k],[epsilon epsilon],'--r')%;
        ylabel('$||e||_2^2$','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        hold off
        subplot(224)
        plot(1:k,Enrm(1:k),'linewidth',1)
        ylabel('Model error','interpreter','latex')
        xlabel('Iteration','interpreter','Latex')
        drawnow
        end
    end
end

if stopit == 0, stopit = maxit; end

m = reshape(m,sqrtn,sqrtn);
if max(abs(m_true))~=0, out.Enrm = Enrm; end
out.diffm = diffm;
out.stopit = stopit;
out.Rnrm2 = Rnrm2;
out.phi = phi;
out.beta = beta;



function [L, L_bar] = discr_grad(n)
%
% D1 = discr_grad(n)
%
% Builds finite difference approximation matrix for the 2-dimensional
% gradient; we assume zero boundary conditions.
%
% Input:  n = number of grid points for x and y
% Output: L = matrix

% Silvia Gazzola, University of Bath
% Ali Gholami, University of Teheran
% April, 2022

% 1-dimensional finite difference approximation of the first derivative
% operator (eq. (2.1))
D = -speye(n) + spdiags(ones(n,1),1,n,n); D(n,n)=0;

% 2-dimensional finite difference approximation of the gradient (eq. (2.3))
Nabla_x = kron(D,speye(n));
Nabla_z = kron(speye(n),D);
L = [Nabla_x; Nabla_z];

L_bar = [Nabla_x, sparse(n^2,n^2)
         sparse(n^2,n^2), Nabla_z];
     

function [p0, MAD, MED]= zscore(p, a)
%
% [p0, MAD, MED]= zscore(p, a)
%
% Returns the maximum value of the normal components of a one-dimensional 
% array p
%
% Input:  p  = one-dimensional array whose maximum of the normal components
%              should be computed 
%         a  = thereshold for defining the normal components
% Output: p0 = maximum of the normal components of p
%         MAD= median absolute deviation of p
%         MED= median of p

% Silvia Gazzola, University of Bath
% Ali Gholami, University of Teheran
% April, 2022

p = sort(abs(p));
p = p(fix(end/2):end);
MAD = 1.4826*(mad(p,1)+eps);
MED = median(p);
z = (p - MED)/(MAD);
idx = find(abs(z) < a);
p0 = max(p(idx));     