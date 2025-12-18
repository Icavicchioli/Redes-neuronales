clc
%% Datos físicos
g   = 9.81;

% Geometría
V1 = 0.5;          % m^3  (500 L)
D1 = 0.66;         % m
A1 = pi*(D1/2)^2;

V2 = 0.1;          % m^3  (100 L)
D2 = 0.40;         % m
A2 = pi*(D2/2)^2;

h1_max = V1/A1;
h2_max = V2/A2;

%% Parámetros hidráulicos
k12 = 5e-4;
k2  = 1e-3;

%% Caudal deseado de salida
q2_star = 0.002;   % m^3/s  (2 L/s)

%% Altura de equilibrio en el tanque 2
h2_star = (q2_star/k2)^2 / (2*g);

%% Caudal nominal requerido
u_star = k2 * sqrt(2*g*h2_star); % si se hacen los cálculos se nota que en equulibrio todos los caudales son iguales, así que este resultado tiene sentido

%% Altura en tanque 1 en equilibrio
h1_star = h2_star + (u_star/k12)^2/(2*g);

%% Mostrar resultados
fprintf('--- RESULTADOS ---\n');
fprintf('h1_max  = %.3f m\n', h1_max);
fprintf('h2_max  = %.3f m\n\n', h2_max);

fprintf('h2*     = %.3f m\n', h2_star);
fprintf('u*      = %.6f m^3/s (%.3f L/s)\n', u_star, u_star*1000);
fprintf('h1*     = %.3f m\n', h1_star);

