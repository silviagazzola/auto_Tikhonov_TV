% This script can be used to test the automatic algorithm for setting the
% balancing parameter in noise-constrained Tikhonov-TV regularization. It
% runs an image denoising and a tomography test problem.

% Silvia Gazzola, University of Bath
% Ali Gholami, University of Teheran
% April, 2022

clear, clc, close all

%% IMAGE DENOISING
% generating the true image
n = 512;

[t1, t2]=meshgrid(linspace(-1,1,n));
xs =(t1.^2 + t2.^2);

xr = zeros(n);
for i=1:n
    for j=1:n
        r = sqrt((i - n/2)^2 + (j - n/2)^2);
        if (r<160)&&(r>=100)
            xr(i,j) = 1;
        elseif (r<100)&&(r>=40)
             xr(i,j) = 2;
        elseif r<40
             xr(i,j) = 2.75 + 1 - r/22.5;
        end
    end
end
m_true = xr+xs;

m_true = m_true - min(m_true(:));
m_true = m_true/max(m_true(:))*255;

% generating the data (adding noise to the true image)

rng(0) %%%
noise = 20*randn(n);
d = m_true + noise;

% forward operator
G = speye(n*n); 
epsilon = norm(noise(:))^2;

% Lagrange penalty parameters
mu(1) = 1; mu(2) = 1; mu(3) = 1;

maxit = 200;

options.fullmatrix = 'on';
options.beta0 = 1;
options.m_true = m_true;
options.NoStop = 'on';
options.StopThr = 1e-4;
options.plotty = 'off'; % 'on'; % 
options.type = 'TT';
options.beta = 'adapt';
options.zthr = 2.5;

[mtt, outtt] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options);

options.beta = 1e2;
[mttvar, outttvar] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options);

% comparison with TV regularization

options.type = 'TV';
[mtv, outtv] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options);

% comparison with Tikhonov regularization
options.type = 'T'; 
[mt, out] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options);

%% TOMOGRAPHY

n = 320;
maxit = 100; % 600; % 100; % 

options.angles = -90:2:90;
[G, d, m_true, PbInfo] = PRtomo(n,options);

optdp.NoiseLevel = 1e-2; % noise level estimate
d = PRnoise(d,optdp.NoiseLevel);

epsilon = (optdp.NoiseLevel*norm(d))^2;

% Lagrange penalty parameters
mu(1) = 50; mu(2) = 0.1; mu(3) = 0.1;

options.fullmatrix = 'off';
options.beta0 = 1e2;
options.m_true = m_true;
options.NoStop = 'on';
options.StopThr = 1e-4;
options.plotty = 'off'; % 'on'; % 
options.type = 'TT';
options.beta = 'adapt';
options.zthr = 2.5;

[mtt, outtt] = automatic_Tikhonov_TV(G, d, epsilon, mu, maxit, options);