function At = compute_deriv4( t, A, dz, D1, D2, g0, dg0, g1, dg1 );
% Solves  LP = -(A^2)_z using FOURTH ORDER finite difference method
% Then computes A_t = AP
% Operator \mathcal{L}: \mathcal{L}(P) = AP - (A^2 P_z)_z
% Finite Difference Version: 
% L(P) + b = [ M_1 - M_2 D_2 - M_3 D_1 ]P - [ M_2 b_2 + M_3 b_1 ]
% Then P = [ -(A^2)_z - b] \ L
    Nz = length(A);
    A2 = A.*A;
    P0 = dg0(t)./g0(t);
    P1 = dg1(t)./g1(t);
% Construct b_1 for the approximation of P_z
    b1  = zeros(Nz,1); 
    b1(1)   = -1/(4*dz)*P0;      b1(2)     =  1/(12*dz)*P0;
    b1(end) =  1/(4*dz)*P1;      b1(end-1) = -1/(12*dz)*P1;

% Construct b_2 for the approximation of P_{zz}
    b2  = zeros(Nz,1); 
    b2(1)   =  11/(12*dz^2)*P0;   b2(2)     = -1/(12*dz^2)*P0;
    b2(end) =  11/(12*dz^2)*P1;   b2(end-1) = -1/(12*dz^2)*P1;

% Construct M_1, a matrix with A on its diagonal
    M1  = spdiags(A,[0],Nz,Nz); 
% Construct M_2, a matrix with A^2 on its diagonal
    M2  = spdiags(A2,[0],Nz,Nz); 
% Construct M_3, a matrix with FD approximation to (A^2)_z on its diagonal
    bA  = zeros(Nz,1); 
    bA(1)   = -1/(4*dz)*(g0(t))^2;      bA(2)     =  1/(12*dz)*(g0(t))^2;
    bA(end) =  1/(4*dz)*(g1(t))^2;      bA(end-1) = -1/(12*dz)*(g1(t))^2;
    A2z = D1*A2+bA;
%     A2z         =  zeros(Nz,1);%D1*A2; 
%     A2z(2:Nz-1) = (-[A2(4:Nz); g1(t)^2] +8*A2(3:Nz)-...
%                    8*A2(1:Nz-2) +[g0(t)^2; A2(1:Nz-3)])/(12*dz);
%     A2z(1)      = [-3 -10 18 -6 1]*[ g0(t)^2; A2(1:4) ]/(12*dz);
%     A2z(Nz)     = [-1  6 -18 10 3]*[A2(Nz-3:Nz); g1(t)^2 ]/(12*dz);
    M3  = spdiags(A2z,[0],Nz,Nz); 

% Construct L and the rhs
    L   = M1-M2*D2-M3*D1;
    rhs = -A2z + M2*b2 + M3*b1;

% Solve system for P; generate A_t
P = L\rhs;
At  = A.*P;