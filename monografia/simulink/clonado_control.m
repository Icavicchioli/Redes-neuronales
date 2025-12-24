%% Extraer señales
y = out.Y(:);        % salida de la planta
r = out.ref(:);      % referencia
e = out.error(:);    % error

N = length(y);


%% Parámetros
na = 10;                 % retardos de salida
n0 = na + 1;
Ns = N - na;

%% Inicialización
X = zeros(na + 2, Ns);   % [y(k-1..k-10); r(k); e(k)]
T = zeros(1, Ns);        % y(k)

for k = n0:N
    % Salidas pasadas
    X(1:na, k-na) = y(k-1:-1:k-na);

    % Entradas actuales
    X(na+1, k-na) = r(k);
    X(na+2, k-na) = e(k);

    % Salida objetivo
    T(k-na) = y(k);
end


%% Crear red
numNeuronas = 20;

net = fitnet(numNeuronas, 'trainlm');

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

% División de datos
net.divideMode = 'sample';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Función de costo
net.performFcn = 'mse';


%% Entrenar
[net, tr] = train(net, X, T);

