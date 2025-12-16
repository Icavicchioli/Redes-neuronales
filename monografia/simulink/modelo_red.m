% La red estima y(n+1) usando y(n ... n-ny+1) y u(n)

y = out.Y(:);   % asegurar vector columna
u = out.u(:);

ny = 10;    % retardos de salida
nu = 1;     % retardos de entrada

N = length(y);
Ns = N - max(ny,nu) - 1;

X = zeros(Ns, ny + nu);
T = zeros(Ns, 1);

for k = 1:Ns
    idx = k + max(ny,nu);

    X(k,1:ny)      = y(idx:-1:idx-ny+1).';
    X(k,ny+1:end)  = u(idx:-1:idx-nu+1).';

    T(k) = y(idx+1);
end
%% ajustamos por media y varianza paara ayudar al entrenamiento, son números muy chicos
% X entra a la red, T sale, es lo que buscamos
muX  = mean(X,1);
sigX = std(X,0,1);
sigX(sigX==0) = 1;   % protección

muT  = mean(T);
sigT = std(T);
if sigT == 0
    sigT = 1;
end

Xn = (X - muX) ./ sigX;
Tn = (T - muT) ./ sigT;
%% entrenamos una red de 20 perceptrones en capa oculta
net = fitnet(20);

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Desactivar preprocesado interno 
net.inputs{1}.processFcns  = {};
net.outputs{2}.processFcns = {};

net = train(net, Xn.', Tn.');

%% Guardado de las variables para usar después

save net_1.mat net muX sigX muT sigT ny nu

%% Simulink

clear net
load net_1.mat

gensim(net)


