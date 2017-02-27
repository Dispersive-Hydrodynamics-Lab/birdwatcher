function At = compute_deriv4_periodic( t, A, dz, D1, D2);
% Solves  LP = -(A^2)_z using FOURTH ORDER finite difference method
% Then computes A_t = AP
% Operator \mathcal{L}: \mathcal{L}(P) = AP - (A^2 P_z)_z
% Domain: Periodic { A(0,t)=A(L,t) => P(0,t) = P(L,t) 
% Finite Difference Version: 
% L(P) = [ M_1 - M_2 D_2 - M_3 D_1 ]P 
% Then P = [ -(A^2)_z ] \ L
    Nz = length(A);
    A2 = A.*A;
% Construct M_1, a matrix with A on its diagonal
    M1  = spdiags(A,[0],Nz,Nz); 
% Construct M_2, a matrix with A^2 on its diagonal
    M2  = spdiags(A2,[0],Nz,Nz); 
% Construct M_3, a matrix with FD approximation to (A^2)_z on its diagonal
    A2z =  zeros(Nz,1);%D1*A2; 
    A2z = ( -circshift(A2,[-2,0]) +...
           8*circshift(A2,[-1,0])  -...
           8*circshift(A2,[ 1,0]) +...
             circshift(A2,[ 2,0]))/(12*dz);
    M3  = spdiags(A2z,[0],Nz,Nz); 

% Construct L and the rhs
    L   = M1-M2*D2-M3*D1;
    rhs = -A2z;

% Solve system for P; generate A_t
P = L\rhs;
At  = A.*P;