%% modelo_red.m
% La red estima y(n+1) usando y(n ... n-ny+1) y u(n)

y = out.Y(:);   % asegurar vector columna
u = out.u(:);

ny = 10;
nu = 1;

N  = length(y);
k0 = max(ny, nu);
Ns = N - k0 - 1;

X = zeros(Ns, ny + nu);
T = zeros(Ns, 1);

for k = 1:Ns
    t = k + k0;               % instante actual
    
    % salidas pasadas: y(t), y(t-1), ..., y(t-ny+1)
    X(k, 1:ny) = y(t:-1:t-ny+1);
    
    % entradas pasadas: u(t), ..., u(t-nu+1)
    X(k, ny+1:end) = u(t:-1:t-nu+1);
    
    % salida futura exacta
    T(k) = y(t+1);
end

%% chequeo de alineación: T adelanta 1 muestra

k0 = max(ny, nu);
Ns = length(T);

figure
plot(y(k0+1:k0+Ns), 'k', 'LineWidth', 1.5)
hold on
plot(T, 'r--', 'LineWidth', 1.5)
legend('y(t)','T = y(t+1)')
grid on

%% normalización por media y varianza

muX = mean(X, 1);
sigX = std(X, 0, 1);
sigX(sigX == 0) = 1;      % protección

muT = mean(T);
sigT = std(T);
if sigT == 0
    sigT = 1;
end

Xn = (X - muX) ./ sigX;   % muestras × entradas
Tn = (T - muT) ./ sigT;   % muestras × 1

%% entrenamos redes con distintas cantidades de neuronas ocultas

% 20 neuronas
net20 = entrenar_red(Xn, Tn, 20);
guardar_red('net_20.mat', net20, muX, sigX, muT, sigT, ny, nu);
cargar_red('net_20.mat');

% 10 neuronas
net10 = entrenar_red(Xn, Tn, 10);
guardar_red('net_10.mat', net10, muX, sigX, muT, sigT, ny, nu);
cargar_red('net_10.mat');

% 5 neuronas
net5 = entrenar_red(Xn, Tn, 5);
guardar_red('net_5.mat', net5, muX, sigX, muT, sigT, ny, nu);
cargar_red('net_5.mat');

% 2 neuronas
net2 = entrenar_red(Xn, Tn, 2);
guardar_red('net_2.mat', net2, muX, sigX, muT, sigT, ny, nu);
cargar_red('net_2.mat');

%% perceptrón lineal (sin capa oculta)

net_lin = entrenar_perceptron_simple(Xn, Tn);
guardar_red('net_lin.mat', net_lin, muX, sigX, muT, sigT, ny, nu);
cargar_red('net_lin.mat');
