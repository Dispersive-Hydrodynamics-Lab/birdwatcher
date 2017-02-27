% Use Jianke's solitary wave algorithm, biconjugate gradients, to
% compute conduit solitons 
function [ z,phi,c ] = conduit_soliton_newton_cg( amplitude )

debug = 0; % Set to 1 to see debugging information, plots, etc.


%%%%% NICOLE: You can change things here %%%%%%%
% Amplitude deviation from unity (now an input for the function)
% Set up Fourier grid
N = 2^13; % Increase the exponent to get better accuracy
          % Decrease the exponent to get faster code
L = 50;  % This code will compute a soliton on the domain [-L, L].
%%%%%%% END CHANGING THINGS %%%%%%%%%%%%%%%


% Total amplitude (background plus soliton height)
Amplitude = 1+amplitude;

% Speed
c = (2*Amplitude^2*log(Amplitude)-Amplitude^2+1)/(Amplitude-1)^2;

if debug
    disp(['Solving for conduit soliton using Newton, ',...
      'biconjugate gradient iterations.']);
    disp(['a = ',num2str(amplitude),', c = ',num2str(c)]);
    tic;
end

% Set up Fourier grid
dz = 2*L/N;
z = dz*[-N/2:N/2-1]';
dk = pi/L;
k = fftshift(dk*[-N/2:N/2-1]');
mk2 = -k.*k;
ik = 1i*k;
Minvhat = 1./(2+c*(mk2-1));

% Initial guess as column vectors
phi0 = amplitude*exp(-(z/5).^2); % Deviation from unit background
phi = phi0;
% phi0 = phi; % Poor man's continuation

% Solver parameters
maxit = 50;
maxlinit = 50;
rtol = 1e-13;
atol = 1e-13;
cgtol = 1e-2;

% Setup
phihat0 = fft(phi0);
ddphi0 = ifft(mk2.*phihat0,'symmetric');
dphi0 = ifft(ik.*phihat0,'symmetric');
Res0 = (2-c)*phi0 + c*(1+phi0).*ddphi0 + phi0.*phi0 - ...
       c*dphi0.*dphi0;
Res0hat = fft(Res0);
errnewton0 = sqrt(abs(sum(Res0.*ifft(Minvhat.*Res0hat,'symmetric'))));

if debug
    %Set up plot
    figure(1);clf
    h=plot(z,phi,'b-');
    axis([z(1),z(end),0.8,Amplitude+0.1]);
end

for ii=1:maxit
  % Check error
  phihat = fft(phi);
  dphi = ifft(ik.*phihat,'symmetric');
  ddphi = ifft(mk2.*phihat,'symmetric');
  Res = (2-c)*phi + ...
        c*(1+phi).*ddphi + ...
        phi.*phi - ...
        c*dphi.*dphi;
  errnewton = sqrt(abs(sum(Res.*ifft(Minvhat.*fft(Res),'symmetric'))));
  if debug
      disp(['ii = ',int2str(ii),', residual = ',num2str(errnewton)]);  
      if errnewton < atol + rtol*errnewton0
          disp('breaking out of Newton iterations');
          break;
      end
      set(h,'ydata',1+phi);
      drawnow;
      %input('continue?');
  elseif errnewton < atol + rtol*errnewton0
          break;
  end
  % Linear solve with cg
  u = zeros(N,1);
  R0 = -Res;
  R = R0;
  Rt = R0;
  Rhat = fft(R);
  D = ifft(Minvhat.*Rhat,'symmetric');
  Dt = D;
  MinvR = ifft(Minvhat.*Rhat,'symmetric');
  RtRInner = sum(Rt.*MinvR);

  % Precompute variable coefficients in linear operators
  L11 = c*(1+phi); L12 = -2*c*dphi; L13 = 2*(1+phi)-c*(1-ddphi);
  L1t1 = L11; L1t2 = -2*L12; L1t3 = 2*(1+phi)-c*(1-4*ddphi);

  if debug
      disp(['     Commencing inner, linear iterations']);
  end
  for jj=1:maxlinit
      Dhat = fft(D);
      L1D = L11.*ifft(mk2.*Dhat,'symmetric') + ...
            L12.*ifft(ik.*Dhat,'symmetric') + ...
            L13.*D;
      a = RtRInner/sum(Dt.*L1D);
      u = u + a*D;
      R = R - a*L1D;
      Dthat = fft(Dt);
      Rt = Rt - a*(L1t1.*ifft(mk2.*Dthat,'symmetric') + ...
                   L1t2.*ifft(ik.*Dthat,'symmetric') + ...
                   L1t3.*Dt);
      Rhat = fft(R);
      MinvR = ifft(Minvhat.*Rhat,'symmetric');
      
      % Check error
      errlinear = sqrt(abs(sum(R.*MinvR)));
      if debug
          disp(['     jj = ',int2str(jj),', residual = ',num2str(errlinear)]);
          if errlinear < cgtol*errnewton
              disp(['     breaking out of linear iterations']);
              break
          end
      elseif errlinear < cgtol*errnewton
              break
      end

      % Continue
      RtRInnerPrev = RtRInner;
      RtRInner = sum(Rt.*MinvR);
      b = RtRInner/RtRInnerPrev;
      D = MinvR + b*D;
      Dt = ifft(Minvhat.*fft(Rt),'symmetric') + b*Dt;
  end
  % Newton update
  phi = phi + u;
end
% phi = 1+phi;
if debug
    toc
end

if debug
    % Plot result
    figure(1)
    clf()
    subplot(2,1,1);
    plot(z,phi,'b-');
    xlabel('$z$','interpreter','latex');
    ylabel('$A$','interpreter','latex');
    axis([z(1),z(end),0.8,Amplitude+0.1]);
    % Fourier coefficients
    subplot(2,1,2);
    phihat = fft(phi);
    phihat = phihat/max(abs(phihat));
    semilogy(k(1:N/2),abs(phihat(1:N/2)),'b-');
    xlabel('$k$','interpreter','latex');
    ylabel('$|\hat{\phi}|/\max |\hat{\phi}|$','interpreter','latex');
    %axis([z(1),z(end),0.8,Amplitude+0.1]);

    figure(2)
    clf()
    plot(z,1+phi,'b-',...
         z,1+amplitude*exp(-z.^2/(4*log(1+amplitude))),'r--');
    xlabel('$z$','interpreter','latex');
    ylabel('$A$','interpreter','latex');
    axis([z(1),z(end),0.8,Amplitude+0.1]);
end