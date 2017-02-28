% Driver script for running the conduit equation solver
% conduit_solver.m
function driver_conduit_solver_will(iter, worknum, zmax, tmax, domain, area)
save_on  = 1;  % Set to nonzero if you want to run the solver, set
                % to 0 if you want to plot
periodic = 1; % set to nonzero to run periodic solver (no BCs need)
              % set to 0 to run solver with time-dependent BCs                

%% Numerical Parameters
numout   = round(tmax);           % Number of output times
t        = linspace(0,tmax,numout);  % Desired output times
dzinit =  1/100; % Spatial Discretization: for most accurate runs
                  % With O(h^4), 0.1 gives 10^{-3} max error over t= [0,53]
Nz       = round(zmax/dzinit);
if periodic
    dz       = zmax/Nz;    % Spatial  discretization
else
    dz       = zmax/(Nz+1);    % Spatial  discretization
end
    h        = 4   ;           % Order of method used     

%% PDE Initial and Boundary Conditions
f = @(z) interp1(domain,area,z,'spline',1);
        ic_type = ['soligas_iter_',num2str(iter)];
    if periodic
        bc_type = 'periodic';
    else
        bc_type = 'time_dependent';
    end

%% Create directory run will be saved to
data_dir = ['./data/conduit_eqtn',...
            '_wnum_',  num2str(worknum),...
            '_tmax_',  num2str(round(tmax)),...
            '_zmax_', num2str(round(zmax)),...
            '_Nz_',   num2str(Nz),...
            '_order_',num2str(h),...
            '_init_condns_',ic_type,...
            '_bndry_condns_',bc_type,...
            '/'];
% Create the data directory if necessary
if ~exist(data_dir,'dir')
    mkdir(data_dir);
else
    disp(['Warning, directory ',data_dir]);
    disp('already exists, possibly overwriting data');
end

savefile = sprintf('%sparameters.mat',data_dir);

%% If chosen, run the solver using the parameters and conditions above
if save_on
    % Load initial data
      zplot  = dz*[1:Nz];
      tplot  = linspace(0,tmax,floor(tmax*10));
      A_init = f(zplot);
    
    if periodic
    % Save parameters
        save(savefile,'t','Nz','dz','zmax','f','periodic', 'worknum');
    % Run timestepper
        conduit_solver_periodic( t, zmax, Nz, h, f, data_dir );      
    else    
    % Save parameters
        save(savefile,'t','Nz','dz','zmax','g0','dg0','g1','dg1','f','periodic');
    % Run timestepper
        conduit_solver( t, zmax, Nz, h, g0, dg0, g1, dg1, f, data_dir );
    end
else
    load(savefile);
end


