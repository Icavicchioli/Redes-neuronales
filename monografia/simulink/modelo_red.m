% La red estima y(n+1) usando y(n ... n-ny+1) y u(n)

y = out.Y(:);   % asegurar vector columna
u = out.u(:);

ny = 10;
nu = 1;

N = length(y);
k0 = max(ny,nu);

Ns = N - k0 - 1;

X = zeros(Ns, ny + nu);
T = zeros(Ns,1);

for k = 1:Ns
    t = k + k0;          % instante actual

    % salidas pasadas: y(t), y(t-1), ..., y(t-ny+1)
    X(k,1:ny) = y(t:-1:t-ny+1);

    % entradas pasadas: u(t), ..., u(t-nu+1)
    X(k,ny+1:end) = u(t:-1:t-nu+1);

    % salida futura exacta
    T(k) = y(t+1);
end
%% vemos si están bien alineados, T adelanta 1 muestra
ny = 10;
nu = 1;
k0 = max(ny,nu);

Ns = length(T);

figure
plot(y(k0+1:k0+Ns),'k','LineWidth',1.5)
hold on
plot(T,'r--','LineWidth',1.5)
legend('y(t)','T = y(t+1)')
grid on

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

%% entrenamos una red de 10 perceptrones en capa oculta
net = fitnet(10);

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

%% entrenamos una red de 5 perceptrones en capa oculta
net = fitnet(5);

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

%% entrenamos una red de 2 perceptrones en capa oculta
net = fitnet(2);

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
%% perceptron sin nada xque no confio
% Red lineal pura (1 perceptrón)
net = feedforwardnet([]);   % SIN capas ocultas

net.layers{1}.transferFcn = 'purelin';

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

net.inputs{1}.processFcns  = {};
net.outputs{1}.processFcns = {};

net = train(net, Xn.', Tn.');

save net_lin.mat net muX sigX muT sigT ny nu

clear net
load net_1.mat

gensim(net)