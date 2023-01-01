function Hout = kuramoto(n,kappa,beta)
% New: kuramoto(preset) and kuramoto(n,kappa,beta) are recognized.
%
% Kuramoto.  Kuramoto's model of synchronizing oscillators.
% The model is a system of n ordinary differential equations.
% The k-th equation is
%
%    (d/dt) theta_k = omega_k + (kappa/n) * sum_j(sin(theta_j-theta_k))
%
%    theta_k is the phase of the k-th oscillator,
%    omega_k is the intrinsic frequency of the k-th oscillator,
%    kappa is a scalar coupling parameter.
%
% The equation can also be written in terms of the gradient of a potential
%
%    v = (4/n^2)*(sum_k sum{j>k} sin((theta_j-theta_k)/2))^2)
%
% The Kuramoto equation is
%
%    (d/dt) theta = omega - (kappa*n/2) * grad(v)
%
% Theta is initially distributed uniformly in the interval [0,2*pi].
% Omega may be distr1ibuted either uniformly or normally in an
% interval controlled by the parameter beta.
% 
% The interface has five presets, five sliders and four toggles.
%
% Five radio buttons labeled "preset" provide the five settings for the
% parameters that are discussed in a forthcoming blog post.  
%
% n is the number of oscillators.
%
% kappa is the coefficient of the nonlinear coupling term.
%
% beta is the half-width of an interval containing the omegas.
% If the uniform/random switch is set to "uniform", the omegas are
% distributed uniformly in the interval [1-beta,1+beta] and there
% is no randomness.  If the switch is set to "random", the omegas
% are drawn from a random normal distribution centered around 1 with
% standard distribution beta.  This is the only source of randomness
% in the model.  If beta = 0, all the omegas are equal to one and
% there is no randomness.
%
% width and step affect the display of the solution, but not its values.
%
% width is the standard deviation of the radius of a ring centered on
% the unit circle containing the display of the angular velocities
% r*exp(i*theta). These velocities are color coded with blues for
% the fastest, yellows for the slowest, and greens in between.
% If width = 0, the velocities are confined to the unit circle.
%
% step controls how often the display is updated.  Increasing step
% increases the speed of the simulation, but does not affect its
% accuracy.  If step = 0, the simulation is paused. 
%
% An arrow shows the "order paramter", the length |z| of
%
%   |z|*exp(i*psi) = 1/n*sum_k(exp(i*theta_k))
%
%   |z| = 0 indicates no synchronization.
%   |z| = 1 is complete synchronization.
%
% rotate is a frame of reference rotating with angular velocity psi.
%
% restart repeats the simulation using the current parameters.
%
% "uniform/random" sets the distribution of omega.
%
% "order/potential" switches between display of the order parameter |z|
% and the potential v.
%
%   v = 1 indicates no synchronization.
%   v = 0 is complete synchronization.
% Links:
% Wikipedia, Kuramoto model, https://en.wikipedia.org/wiki/Kuramoto_model.
%
% Dirk Brockman and Steven Strogatz, "Ride my Kuramotocycle",
% https://www.complexity-explorables.org/explorables/ride-my-kuramotocycle.
%
% Cleve Moler, "Kuramoto Model of Synchronized Oscillators",
% <https://blogs.mathworks.com/cleve/2019/08/26/kuramoto-model-of-synchronized-oscillators
%
% Cleve Moler, "Experiments With Kuramoto Oscillators", 
% https://blogs.mathworks.com/cleve/2019/09/16/experiments-with-kuramoto-oscillators.
%
% Cleve Moler, "Stability of Kuramoto Oscillators", 
% https://blogs.mathworks.com/cleve/2019/10/30.
% Copyright 2019 Cleve Moler
% Copyright 2019 The MathWorks, Inc.
 
    if nargin == 1
        preset = n;
    else
        preset = 0;
    end
    if nargin == 0
        n = 5;
        kappa = rand;
        beta = .660*kappa;
    elseif nargin < 3
        n = [];       % Number of oscillators.
        kappa = [];   % Coupling coefficient.
        beta = [];    % Half-width of omega interval.
    else % nargin == 3
        % n,kappa,beta are input
    end
    
    width = [];   % Standard deviation of display ring.
    step = [];    % Display step size.
    pot = [];     % Graph potential or order parameter.
    uni = [];     % Uniform or normal distribution of omega.
    quit = [];    % 0 <= t <= quit.
    H = [];       % Optional output
    wanth = nargout > 0;
    delta = [];
    
    init_vars(preset)
    
    point = []; flag = [];  animal = []; titl = [];
    theta = []; omega = []; colors = []; r = [];
    rotate = []; txt = []; ax2 = []; psi = [];
 
    % rng(0)
    stop = 0;
    while ~stop
        init_controls
        init_figure
        init_plot
        shg
        t = 0;      
        loop = 1;
        flag = 0;
        while loop
            s = max(step,.01);
            options = odeset('outputfcn',@outputfcn, ...
                'maxstep',s,'initialstep',s);
            if isinf(quit)
                tspan = t:s:t+delta;
            else
                tspan = t:s:t+quit;
            end
            [t,theta] = ode45(@ode,tspan,theta,options);
            t = t(end);
            theta = theta(end,:)';
        end
    end
    if wanth
        Hout = H;
    end
    close(gcf)
   
%-------------------------------------------------------------
    function theta_dot = ode(~,theta)
        % Kuramoto's system of odes.
        % The nonlinear term is the gradient of the potential.
        % omega and theta are real column vectors of length n.
        % n and kappa are real scalars.
       
        theta_dot = omega - kappa/n*gradv(theta);
    end
    function g = gradv(theta)
        % Gradient of the potential, partial(v)/partial(theta).
        % theta-theta' is a matrix with elements theta(j)-theta(k).
        % The sum is by colums and produces a column vector.
        g = sum(sin(theta-theta'),2);
    end
    function v = potential(theta)
        % The potential.
        v = 0;
        for k = 1:n
            j = k+1:n;
            v = v + sum(sin((theta(j)-theta(k))/2).^2);
        end
        v = (4/n^2)*v;
    end
    function status = outputfcn(t,theta,odeflag)
        % Called after each successful step of the ode solver.
        
        if isequal(odeflag,'init') && wanth
            % assert(t(1) == 0)
            H.t = 0; 
            H.theta = theta;
            H.pot = potential(theta);
            H.gradv = gradv(theta);
            H.psi = psi;
            H.order = 0-0;
        end
        if isempty(odeflag)  % Not 'init' or 'last'.
            for j = 1:length(t)
                if loop == 0
                    break
                end
                % Order parameter.  
                z = 1/n*sum(exp(1i*theta(:,j)));
                psi = 0;
                if get(rotate,'value') == 1
                    % Rotating frame of reference.
                    psi = angle(z);
                    z = abs(z);
                end
                for k = 1:n
                    set(point(k), ...
                        'xdata',r(k)*cos(theta(k,j)-psi), ...
                        'ydata',r(k)*sin(theta(k,j)-psi))
                end
                % Length of arrow is order parameter.
                arrow(0,z);
                if pot
                    % Potential
                    v = potential(theta(:,j));
                    set(titl,'string', ...
                        ['potential (' sprintf('%6.3f',v) ')'])
                    addpoints(animal,t(j),v)
                else
                    % Order parameter
                    set(titl,'string', ...
                        ['order parameter (' sprintf('%6.3f',abs(z)) ')'])
                    addpoints(animal,t(j),abs(z))
                end
                
                if wanth
                    nt = length(H.t) + 1;
                    H.t(nt) = t(j);
                    H.theta(:,nt) = theta(:,j);
                    H.pot(nt) = potential(theta(:,j));
                    H.gradv(:,nt) = gradv(theta(:,j)); 
                    H.psi(nt) = psi;
                    H.order(nt) = abs(z);
                end
                if isinf(quit), qt = delta; else, qt = quit; end
                if rem(t(j),qt) < 1 && t(j) > 10 || t(j) >= quit
                    clearpoints(animal)
                    set(ax2,'xlim',[t(j)-1 t(j)+qt])
                    if t(j) >= quit
                        loop = 0;
                        stop = 1;
                    end
                end
            end
        end
        status = flag + stop;
        drawnow limitrate
    end
    function init_vars(preset)
        width = 0;   % Standard deviation of display ring.
        step = 0.1;  % Display step size.
        pot = true;  % Graph potential or order parameter.
        uni = true;  % Uniform distribution of omega.
        quit = inf;  % 0 <= t <= quit.
        delta = 200*(1+double(preset==3));
        % delta = 800;
        if wanth
            quit = delta;
        end
        
        if preset > 0
            n = 5;
        end
        switch preset
            case 0
            case 1
                kappa = 0;
                beta = 0;
            case 2
                kappa = .75;  % Synchronizes at about t = 85.
                beta = 0;
            case 3
                kappa = .36;
                beta = .24;
            case 4
                kappa = .36;
                beta = .23;
            case 5
                uni = false;  % Random normal omega.
                n = 100;  
                kappa = .10;
                beta = .05;
                step = 0.5;
                width = .05;
            otherwise
                display(['unknown preset:' int2str(preset)])
                scream
        end
        if wanth
            H.n = n;
            H.kappa = kappa;
            H.beta = beta;
            H.t = [];
            H.theta = [];
            H.pot = [];
            H.gradv = [];
            H.psi = [];
            H.order = [];
        end
        loop = 0;
    end
    function txtval(v,vmin,vmax,fmt,cb,k)
        % Slider with text.
        txt(k) = uicontrol('style','text', ...
            'string',sprintf(fmt,v), ...
            'units','normal', ...
            'position',[.04 0.92-.10*k .18 .05], ...
            'background',[.94 .94 .94], ...
            'fontsize',get(0,'defaultuicontrolfontsize')+2, ...
            'horiz','left', ...
            'background','w');
        uicontrol('style','slider', ...
            'units','normal', ...
            'position',[.04 0.88-.10*k .18 .05], ...
            'min',vmin, ...
            'max',vmax, ...
            'value',v, ...
            'callback',cb);
    end
    function t = toggle(str,v,cb,k,lr)
        % Toggle switches.
        switch lr
            case 'l'  % left side
                x = .04;
                dx = .18;
                y = .34;
                dy = .07;
            case 'r'  % right side
                x = .90;
                dx = .08;
                y = .98;
                dy = .08;
        end
        t = uicontrol('style','toggle', ...
            'units','normal', ...
            'position',[x y-k*dy dx dy-.02], ...
            'string',str, ...
            'value',v, ...
            'callback',cb);
    end
        
    function init_controls
        % Initialize buttons, sliders and toggles.
        shg
        % Preset buttons
        uicontrol('style','text', ...
            'string','preset', ...
            'units','normal', ...
            'horiz','left', ...
            'background','w', ...
            'position', [.04 .92 .10 .04])
        for k = 1:5
            uicontrol('style','radio', ...
                'units','normal', ...
                'position',[.04*k .88 .04 .04], ...
                'background','w', ...
                'value',double(k == preset), ...
                'callback',@radiocb)
        end
        
        % Sliders
        txt = zeros(5,1);
        txtval(n,1,100,'n = %3d',@ncb,1);
        txtval(kappa,0,1.0,'kappa = %5.3f',@kappacb,2);
        txtval(beta,0,1.0,'beta = %5.3f',@betacb,3);
        txtval(step,0,1,'step = %5.3f',@stepcb,4);
        txtval(width,0,0.2,'width = %5.3f',@widthcb,5);
                
        % Toggles
        toggle('restart',0,@restartcb,1,'l');
        rotate = toggle('rotate',1,[],2,'l');
        if uni
            toggle('uniform / random',0,@unicb,3,'l');
        else
            toggle('random / uniform',1,@unicb,3,'l');
        end
        if pot
            toggle('potential / order',1,@potcb,4,'l');
        else
            toggle('order / potential',0,@potcb,4,'l');
        end
        toggle('exit',0,@stopcb,1,'r');
        toggle('help',0,@helpcb,2,'r');
        toggle('blog',0,@blogcb,3,'r');
        flag = 0;
        stop = 0;
    end
    function init_figure(~)
        % Initialize figure window.
        set(gcf,'menubar','none','numbertitle','off', ...
             'name','kuramoto','color','white')
        
        ax1 = axes('position',[.30 .34 .60 .60]);
        circle = exp((0:.01:1)*2*pi*1i);
        line(real(circle),imag(circle),'color',grey)
        axis(1.2*[-1 1 -1 1])
        axis square
        set(gca,'xtick',[],'ytick',[])
        box on
        
        ax2 = axes('position',[.3 .07 .6 .2]);
        animal = animatedline('linewidth',2, ...
            'color',cyan);
        if isinf(quit)
            axis([0 delta 0 1.2])
        else
            axis([0 quit 0 1.2])
        end
        titl = title('');
        box on
        
        axes(ax1)
    end
    function init_plot
        % Initialize plot.
        
        % Oscillators initially uniform around circle.
        theta = (1:n)'/n*2*pi;     
        
        omega = omegas(uni);       
        
        % Plot radii.
        r = ones(n,1) + width*randn(n,1);
        
        % Parula, blue is fast, yellow is slow.
        colors = flipud(parula(ceil(n)));
        
        point = zeros(n,1);
        for k = 1:n
           point(k) = line(r(k)*cos(theta(k)),r(k)*sin(theta(k)), ...
                'linestyle','none', ...
                'marker','o', ...
                'markersize',6, ...
                'markeredgecolor','k', ...
                'markerfacecolor',colors(mod(k-1,length(colors))+1,:));
        end
        
        title(sprintf('kappa = %5.3f, beta = %5.3f',kappa,beta))
    end
    function omega = omegas(uni)
        % Intrinsic freqencies in interval of width beta centered at 1.
        if uni
            omega = ones(n,1) + beta*(-1:2/(n-1):1)';
        else
            omega = ones(n,1) + beta*(2*rand(n,1)-1);
        end
    end
    function ncb(arg,~)
        % Number of oscillators.
        n = round(get(arg,'value'));
        set(arg,'value',n)
        set(txt(1),'string',sprintf('n = %d',n))
        loop = 0;
    end
    function kappacb(arg,~)
        % Coupling parameter.
        kappa = get(arg,'value');
        set(txt(2),'string',sprintf('kappa = %5.3f',kappa))
    end
    function betacb(arg,~)
        % Spread of intrinsic frequencies.
        beta = get(arg,'value');
        set(txt(3),'string',sprintf('beta = %5.3f',beta))
        omega = omegas(uni);
    end
    function radiocb(arg,~)
        pos = get(arg,'position');
        preset = pos(1)/.04;
        init_vars(preset)
    end
    function stepcb(arg,~) 
        % Step size for display.
        step = get(arg,'value');
        set(txt(4),'string',sprintf('step = %5.3f',step))
        flag = 1;
    end 
    function widthcb(arg,~)
        % Standard deviation of random radii.
        oldw = width;
        width = get(arg,'value');
        set(txt(5),'string',sprintf('width = %5.3f',width))
        r = (r - 1)*width/(oldw+realmin) + 1;
        flag = 1;
    end
    function restartcb(~,~)
        % Restart with current parameters.
        clearpoints(animal)
        drawnow
        loop = 0;
        %flag = 1;
    end
    function unicb(arg,~)
        uni = get(arg,'value') == 1;
        if uni
            set(arg,'string','uniform / random');
        else
            set(arg,'string','random / uniform');
        end
        omega = omegas(uni);
    end
    function potcb(arg,~)
        pot = get(arg,'value') == 1;
        if pot
            set(arg,'string','potential / order')
        else
            set(arg,'string','order / potential')
        end
    end
    function stopcb(~,~)
        loop = 0;
        stop = 1;
    end
    function helpcb(~,~)
        doc('kuramoto')
        toggle('help',0,@helpcb,2,'r');
    end
    function blogcb(~,~)
        web(['https://blogs.mathworks.com/cleve/2019/10/30' ...
             '/stability-of-kuramoto-oscillators'])
        toggle('blog',0,@blogcb,3,'r');
    end
    function arrow(z0,z1)
        delete(findobj('tag','arrow_shaft'))
        delete(findobj('tag','arrow_head'))
        rho = angle(z1-z0);
        x = real(z1);
        y = imag(z1);
        u = [0 -.08 -.05 -.08 0];
        v = [0 -.05 0 +.05 0];
        s = u;
        u = u*cos(rho) - v*sin(rho) + x;
        v = s*sin(rho) + v*cos(rho) + y;
        line([real(z0) real(z1)],[imag(z0) imag(z1)], ...
            'linewidth',1.5, ...
            'color',cyan, ...
            'tag','arrow_shaft');
        patch(u,v,cyan, ...
            'tag','arrow_head');
    end
            
    function c = grey
        c = [.8 .8 .8];
    end
            
    function c = cyan
        c = [0 .6 .6];
    end
end
