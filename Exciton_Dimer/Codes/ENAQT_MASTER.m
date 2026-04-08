clear all
id = eye(2);
sz = [1 0; 0 -1];
sx = [0 1; 1 0];
sy = [0 -1j; 1j 0];
sm = [0 1; 0 0];
sp = [0 0; 1 0];
V = 1;
E = 1;
n = 4;
g_deph_all = [1e-05, 3.3e-05, 6.7e-05, 1e-04, 3.3e-04, 6.7e-04, 1e-03, 3.3e-03, 6.7e-03, 1e-02, 3.3e-02, 6.7e-02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50];
target=3;
target_state=target_mapping_qubits(target,n); 

Energies = [-0.221202  , -0.1165728 ,  1.6135948 , -0.17796115]; %E + randn(n,1)*E*0;

temp_vect = zeros(n,1);
temp_vect(2) = V;
temp_vect(end) = V;
Hs = toeplitz(temp_vect);
Hs = Hs + diag(sum(Energies)-2*Energies);

traj_tot=1;

psis = zeros(2^n, 1);
psis(end/2+1) = 1;
rhos0=kron(psis,psis');
rhob=id/2;


dt = 0.2;
tf = 40;
times = 0:dt:tf;

enaqt_collision=zeros(1,max(size(g_deph_all)));
enaqt_classical=zeros(1,max(size(g_deph_all)));
enaqt_Lindblad=zeros(1,max(size(g_deph_all)));

cont=1;
for g_deph = g_deph_all
    c_int = sqrt(g_deph/4/dt);
    
    P=zeros(2^n,max(size(times)));
    c=1;
    for time=times
        if time ~=0

            for i=1:n
                rhos = expm(-1j*Energies(i)*kron3(eye(2^(i-1)), sz, eye(2^(n-i)))*dt)*rhos*expm(1j*Energies(i)*kron3(eye(2^(i-1)), sz, eye(2^(n-i)))*dt);
            end
            for i=1:n
                if i~=n
                    for j=i+1:n
                        if j-i==1 || j-i==n-1
                            rhos = expm(-1j*(V/2*kron5(eye(2^(i-1)), sx, eye(2^(j-i-1)), sx, eye(2^(n-j))) + V/2*kron5(eye(2^(i-1)), sy, eye(2^(j-i-1)), sy, eye(2^(n-j))))*dt)*rhos*expm(1j*(V/2*kron5(eye(2^(i-1)), sx, eye(2^(j-i-1)), sx, eye(2^(n-j))) + V/2*kron5(eye(2^(i-1)), sy, eye(2^(j-i-1)), sy, eye(2^(n-j))))*dt);
                        end
                    end
                end
            end
            for i=1:n
                rho = expm(-1j*(c_int*kron4(eye(2^(i-1)), sz, eye(2^(n-i)), sz)*dt))*kron(rhos,rhob)*expm(1j*(c_int*kron4(eye(2^(i-1)), sz, eye(2^(n-i)), sz)*dt));
                for iter1=0:2^n-1
                    for iter2=0:2^n-1
                        rhos(iter1+1,iter2+1) = trace(rho(iter1*2+1:iter1*2+2,iter2*2+1:iter2*2+2));
                    end
                end

            end
        else
            rhos=rhos0;
        end
        
        P(:,c)=diag(rhos);
        c=c+1;
    end

    enaqt_collision(cont)=sum(P(target_state,:));

    %%
    omega=sqrt(g_deph);

    for traj=1:traj_tot
        psi = zeros(n,1);
        psi(1)=1;

        c=1;
        P_c = zeros(n,max(size(times)));
        for time=times
            psi = expm(-1j*(Hs*dt+diag(randn(n,1)*omega)*sqrt(dt)))*psi;
            P_c(:,c)=abs(psi).^2;
            c=c+1;
        end
        enaqt_classical(cont) = enaqt_classical(cont) + sum(P_c(target,:))/traj_tot;
    end
    
    %%
    gamma=2*g_deph;

    L=zeros(n^2,n^2);
    for i=1:n
        for j=1:n
            for l=1:n
                for k=1:n
                    L((i-1)*n+j, (l-1)*n+k) = -1j * (Hs(i, l)*eq(j, k) - Hs(j, k)*eq(i,l)) - gamma/2*eq(i,l)*eq(j,k)*(1-eq(i,j));
                end
            end
        end
    end

    rhos = zeros(n^2,1);
    rhos(1)=1;

    U=expm(L*dt);

    c=1;
    P_L = zeros(n, max(size(times)));

    for time=times
        rhos = U*rhos;
        P_L(:,c)=diag(reshape(rhos, n, n));
        c=c+1;
    end

    enaqt_Lindblad(cont) = sum(P_L(target,:));

    %%
    cont=cont+1;

end

figure
plot(g_deph_all,enaqt_Lindblad*dt, 'DisplayName','Exact numerical solution')
hold on
plot(g_deph_all,enaqt_classical*dt, 'DisplayName','Classical noise algorithm')
hold on
plot(g_deph_all,enaqt_collision*dt, 'DisplayName','Collision algorithm')
legend
xlabel('$\frac{g}{V}$', 'Interpreter', 'latex')
ylabel(['$\eta_' num2str(target) '(' num2str(tf) ')$'], 'Interpreter', 'latex')

%%
function matrice=kron3(a,b,c)
    matrice=kron(kron(a,b),c);
end

function matrice=kron4(a,b,c,d)
    matrice=kron(kron(kron(a,b),c),d);
end

function matrice=kron5(a,b,c,d,e)
    matrice=kron(kron(kron(kron(a,b),c),d),e);
end

function target_state = target_mapping_qubits(target_site, n)
    states = zeros(n,1);
    states(1)=2;
    for i=2:n
        states(i) = states(i-1)+2^(i-2);
    end
    target_state = states(n-target_site+1);
end