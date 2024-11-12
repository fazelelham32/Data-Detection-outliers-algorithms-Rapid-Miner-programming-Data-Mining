clc;
clear;
close all;

%% Problem Definition

Method = 'CS';	% DB or CS

data = load('mydata');
X = data.X;
k = 10;

CostFunction=@(s) ClusteringCost(s, X, Method);     % Cost Function

VarSize=[k size(X,2)+1];  % Decision Variables Matrix Size

nVar=prod(VarSize);     % Number of Decision Variables

VarMin= repmat([min(X) 0],k,1);      % Lower Bound of Variables
VarMax= repmat([max(X) 1],k,1);      % Upper Bound of Variables

%% ABC Settings

MaxIt=100;              % Maximum Number of Iterations

nPop=100;               % Population Size (Colony Size)

nOnlooker=nPop;         % Number of Onlooker Bees

L=round(0.5*nVar*nPop); % Abandonment Limit Parameter

a=1;                    % Acceleration Coefficient Upper Bound

%% Initialization

% Empty Bee Structure
empty_bee.Position=[];
empty_bee.Cost=[];
empty_bee.Out=[];

% Initialize Population Array
pop=repmat(empty_bee,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);
    
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end

% Abandonment Counter
C=zeros(nPop,1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% ABC Main Loop

for it=1:MaxIt
    
    % Recruited Bees
    for i=1:nPop
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=unifrnd(-a,+a,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position=max(newbee.Position, VarMin);
        newbee.Position=min(newbee.Position, VarMax);
        
        % Evaluation
        [newbee.Cost, newbee.Out]=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F=zeros(nPop,1);
    for i=1:nPop
        if pop(i).Cost>=0
            F(i)=1/(1+pop(i).Cost);
        else
            F(i)=1+abs(pop(i).Cost);
        end
    end
    P=F/sum(F);
    
    % Onlooker Bees
    for m=1:nOnlooker
        
        % Select Source Site
        i=RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=unifrnd(-a,+a,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position=max(newbee.Position, VarMin);
        newbee.Position=min(newbee.Position, VarMax);
        
        % Evaluation
        [newbee.Cost, newbee.Out]=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Scout Bees
    for i=1:nPop
        if C(i)>=L
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
            [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);
            C(i)=0;
        end
    end
    
    % Update Best Solution Ever Found
    for i=1:nPop
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    % Plot Solution
    figure(1);
    PlotSolution(X, BestSol);
    pause(0.01);

end
    
%% Results

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');

