function conduit_solver_periodic( t, zmax, Nz, h, f, directory );
% Solves the scalar conduit equation using a FD method in space
% and Matlab's ode45 solver in time
% Solves:
%
%  A_t = A*P
%  A*P - (A^2*P_z)_z = -(A^2)_z
%  with periodic BCs, A(z,0) = f(z)
%
% Inputs:
%
% t        :  1D array of output times
% zmax, dz :  spatial grid parameters 
% h        :  order of finite difference method 
% f        :  initial condition in space at t=0
% directory - output directory where data at each output timestep
%             is written to a file conduit#####.mat
%
% Outputs:  NONE except for data written to files
%
  global inc dir tout start;

  % Set global variables for ode solver output function
  tout = t;
  dir = directory;
  inc = 0;
  dt = 1e-1;

  % Setup grid
  dz = zmax/(Nz);
  z = dz*[1:Nz]';


  %
  % Output what we are about to do
  %
  disp(['Solving conduit eqtn.']);
  disp([' Time interval:  [', num2str(t(1)),...
        ',',num2str(t(end)),']']);
  
  %
  % Construct initial condition on spatial domain
  %
  A_init = f(z);
  A      = A_init; tnow = min(t);
  save(strcat(dir,num2str(inc,'%05d')),'A_init','inc');
  inc = inc + 1;
  save(strcat(dir,num2str(inc,'%05d')),'A','tnow','inc');
  disp(['Time = ',num2str(tout(inc)),', inc = ',int2str(inc),...
          '/',int2str(length(tout))]);
  
  %Set appropriate ODE solver options.
  opts = odeset('OutputFcn',@output,'Refine',1,'RelTol',1e-6, ...
                'AbsTol',1e-6,'Stats','on','maxstep',dt);
  start = tic;

  % Get soln using Matlab RK timestepper (myode23 or myode45 both work)
  if h ==2 % Use second order FD method
      %
      % Precompute differentiation matrices 
      %
      D1  = 1/(2*dz) * spdiags([ -ones(Nz,1),...
                                  zeros(Nz,1),...
                                  ones(Nz,1)],...
                               -1:1, Nz,Nz);
      D2  = 1/(dz^2) * spdiags([  ones(Nz,1),...
                               -2*ones(Nz,1),...
                                  ones(Nz,1) ],...
                               -1:1,Nz,Nz);
    ode45(@(t,y) compute_deriv2_periodic( t, y, dz, D1, D2),...
                                 tout, A_init, opts);
  elseif h ==4 % Use fourth order FD method
      %
      % Precompute differentiation matrices 
      %      
      D1  = 1/(dz) * spdiags([[  1/12*ones(Nz,1) ],...
                              [ -2/3 *ones(Nz,1) ],...
                              [      zeros(Nz,1) ],...
                              [  2/3 *ones(Nz,1) ],...
                              [ -1/12*ones(Nz,1) ] ],...
                                -2:2,Nz,Nz);
      D1(1:2,Nz-1:Nz) = 1/(dz) * [ 1/12 -2/3;  0   1/12];
      D1(Nz-1:Nz,1:2) = 1/(dz) * [-1/12   0 ; 2/3 -1/12];
      D2  = 1/(dz^2) * spdiags([[ -1/12 *ones(Nz,1)],...
                                [  4/3  *ones(Nz,1)],...
                                [ -5/2  *ones(Nz,1)],...
                                [  4/3  *ones(Nz,1)],...
                                [ -1/12 *ones(Nz,1)] ],...
                                -2:2,Nz,Nz);
      D2(1:2,Nz-1:Nz) = 1/(dz^2) * [-1/12  4/3;  0  -1/12];
      D2(Nz-1:Nz,1:2) = 1/(dz^2) * [-1/12   0 ; 4/3 -1/12];
    ode45(@(t,y) compute_deriv4_periodic( t, y, dz, D1, D2),...
                                 tout, A_init, opts);
  end
  % Finish and clean up
  finish = toc(start);
  disp('Calculation Finished');
  days_left = datenum([0 0 0 0 0 toc(start)]);
  time_left=datevec(days_left-floor(days_left));
  disp(['Computation time = ',...
        int2str(floor(days_left)),'d ',...
        int2str(time_left(4)),'h ',...
        int2str(time_left(5)),'m ',...
        num2str(time_left(6)),'s']);
  
%
% Called by odesolver at every output time
%
function status = output(t,y,flag)
  global inc dir tout start;
  status = 0;
  if strcmp(flag,'')
    for ii=1:length(t)
      % Extract solution
      A = y(:,ii);
      tnow = t(ii);
      % Increment and output estimate of time left
      inc = inc + 1;
      % Output results to file for subsequent analysis
      save(strcat(dir,num2str(inc,'%05d')),'A','tnow','inc');
      percentage_of_work = (inc)/(length(tout));
      seconds_left = ((1/percentage_of_work)-1)*(toc(start));
      days_left = datenum([0 0 0 0 0 seconds_left]);
      time_left=datevec(days_left-floor(days_left));
      disp(['Time = ',num2str(tout(inc)),...
          ', inc = ',int2str(inc),'/',int2str(length(tout)),...
          ', computation time left = ',...
          int2str(floor(days_left)),'d ',...
          int2str(time_left(4)),'h ',...
          int2str(time_left(5)),'m ',...
          num2str(time_left(6)),'s']);
    end
  end





  