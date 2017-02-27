function At = compute_deriv2( t, A, dz, D1, D2, g0, dg0, g1, dg1 );
% Solves  LP = -(A^2)_z using SECOND ORDER finite difference method
% Then computes A_t = AP
% Operator \mathcal{L}: \mathcal{L}(P) = AP - (A^2 P_z)_z
% Finite Difference Version: 
% L(P) + b = [ M_1 - M_2 D_2 - M_3 D_1 ]P - [ M_2 b_2 + M_3 b_1 ]
% Then P = [ -(A^2)_z - b] \ L
    Nz = length(A);
    A2 = A.*A;
    P0 = dg0(t)/g0(t);
    P1 = dg1(t)/g1(t);
% Construct b_1 for the approximation of P_z
    b1  = zeros(Nz,1); b1(1) = -1/(2*dz)*P0; b1(end) = 1/(2*dz)*P1;

% Construct b_2 for the approximation of P_{zz}
    b2  = zeros(Nz,1); b2(1) = 1/(dz^2)*P0; b2(end) = 1/(dz^2)*P1;

% Construct M_1, a matrix with A on its diagonal
    M1  = spdiags(A,[0],Nz,Nz); %sparse(diag(A));
% Construct M_2, a matrix with A^2 on its diagonal
    M2  = spdiags(A2,[0],Nz,Nz); %sparse(diag(A2));
% Construct M_3, a matrix with FD approximation to (A^2)_z on its diagonal
    A2z      =  zeros(Nz,1);%D1*A2; 
    A2z(2:Nz-1) = (A2(3:Nz)-A2(1:Nz-2))/(2*dz);
    A2z(1)   =  (A2(2) - g0(t)^2)/(2*dz); %A2z(1)   - (g0(t))^2/(2*dz); 
    A2z(Nz) =  (g1(t)^2 - A2(Nz-1))/(2*dz); %A2z(end) + (g1(t))^2/(2*dz); 
    M3  = spdiags(A2z,[0],Nz,Nz); %sparse(diag(A2z));

% Construct L and the rhs
    L   = M1-M2*D2-M3*D1;
    rhs = -A2z + M2*b2 + M3*b1;

% Solve system for P; generate A_t
P = L\rhs;
At  = A.*P;