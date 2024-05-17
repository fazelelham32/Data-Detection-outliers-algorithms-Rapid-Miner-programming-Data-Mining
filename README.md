# PSO algorithm

Genetic Algorithm: a tool for optimization, we have three basic operators: intersection-mutation-selection (at the same time iteratively forms a population)
The weakness of this algorithm:
There is no such thing as information flow and cooperation.
Collective intelligence: the main components that include crowd + cooperation = communication.
Any communication is meaningless without the exchange of information, and the need for an information flow that leads to the existence of cooperation. For this, there is a need for a rule called self-regulation that everyone follows. Wherever these two are, there is also collective intelligence
The human brain is also an example of collective intelligence. A neuron in the human brain takes 1 millisecond to respond to incoming information.
Colony of ants and bees: together they can do things that no one ant can do alone.
The performance of gregarious animals: birds-fishes
Achieving a series of powerful social models with "normal" people by applying collective intelligence roles can achieve collective performance and reach the PSO algorithm which could be used as an optimization.
PSO, we have a series of hypothetical living beings, each of which has a position and a proposal for optimization, these are moving in space, and the law is fixed for all of them. Finally, they can reach the desired optimal solution to the problem and the defined criteria are either maximum or minimum
In particle swarm optimization algorithm: basic model
Xi: position
Vi: Speed
Xi best: the best position experienced
Equations describing the behavior of particles:
Neighborhood: instead of making the best particle the source of my inspiration, they look more locally, neighborhood defines and I define the best particle in the best particle.
Neighborhood has two types of definitions: geometrical or geographical: where the distance is decisive.
Social neighborhood
FIPS Algorithm: Everyone follows each other conditionally. Information is fully circulated.
Binary PSO algorithm: Probability of one bit of the position vector:
One choice for the π function is the sigmoid pan with one argument. The only argument to the link function is the velocity of the particle.
Applications of PSO: types of continuous and discrete optimization problems - modeling and control - processing types of digital signals and pattern recognition - management of power generation and distribution systems - design and optimization of communication networks - forecasting and modeling of economic models - system design automatic and robotic and.
25 years after the genetic algorithm.
Particle properties:
All of the above are dependent.
Definitions related to optimization:
The target pan is the value that we want to optimize, for which we can imagine two situations.
1. Optimization of fitness function and profit function
2. Minimization or minimization of objective function criteria, cost function, cost function, and error function
The condition of breaking a personal record by member I of the population:
 
The condition for breaking the general stagnation by the i-th particle:
 
The position of the ith particle, and its jth component:
 
 
The lower the inertia, the faster the algorithm converges.
C1 and c2 are the personal and collective learning coefficients, respectively.
We have two concepts in optimization algorithms:
Exploration: The ability to generate a new answer, the meaning of which is search, the ability to generate a new answer is increased in the interest of this subject
Exploitation: the ability of current W responses to be reduced in favor of this subject.
If c1 and c2 become very big, then it can help Exploration.
If c1 and c2 become too small, then it can help exploitation.
Steps of PSO algorithm:
1- Creation of the initial population and its evaluation
2- Determining the best personal and collective memories
3- Updating speed and position and evaluating new answers
4- If the stop conditions are not met, we go to step 2.
5- The end
Regarding the stopping conditions, we can determine several types.
1. This is the target point in the diagram.
2. We have the highest number of repetitions.
 
Since we feel that this is almost constant, we start a counter, when the request exceeds a fixed value, we stop this number of records. It is better to start again and solve it or not to solve it again.
Termination conditions that are not specific to any optimum.
1- Reaching an acceptable level of response
2- Passing the number of repetitions or specified time
3- Passing the specified number of repetitions or time without seeing any improvement in the result
4- Checking a certain number of answers
Therefore, in the pos algorithm at time t is equal to:
Npop: population size
 
Time is not a good factor.
Implementation in C# environment for PSO algorithm and we defined various functions for it, for example, restringing function: a function that has many minimum and maximum points. The famous function is the benchmark function.
If we set the particles to the global best, there is a kind of centrality.
If we increase the coefficient of c2 a little, we can see that they accumulate as local optima, that is, whoever is better than others in any field attracts the rest to himself.
If I increase the share of insider learning, they start to fluctuate, other more successful people conflict with insiders.
This is a continuous sphere benchmark function.
Implementation in MATLAB:
The program is divided into several parts: The first part is the definition of the problem. In the second part: we determine the algorithm settings and other parameters, in the third part: program execution and the implementation of program settings, and in the fourth part, we enter the program loop, and in the last part, we process the program.
The best answer we got so far is in the program below GlobalBest.
GlobalBest is a static property.
After the evaluation, we want to check if he broke his record, and then the total record
CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
 
CostFunction=@(x) Sphere(x);        % Cost Function
 
nVar=5;             % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=4000;      % Maximum Number of Iterations
 
nPop=20;        % Population Size (Swarm Size)
 
% w=1;            % Inertia Weight
% wdamp=0.99;     % Inertia Weight Damping Ratio
% c1=2;           % Personal Learning Coefficient
% c2=2;           % Global Learning Coefficient
 
% Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
w=chi;          % Inertia Weight
wdamp=1;        % Inertia Weight Damping Ratio
c1=chi*phi1;    % Personal Learning Coefficient
c2=chi*phi2;    % Global Learning Coefficient
 
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
 
%% Initialization
 
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
 
particle=repmat(empty_particle,nPop,1);
 
GlobalBest.Cost=inf;
 
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
BestCost=zeros(MaxIt,1);
 
nfe=zeros(MaxIt,1);
 
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position)...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    nfe(it)=NFE;
    
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end
 
%% Results
 
figure;
%plot(nfe,BestCost,'LineWidth',2);
semiology(nfe,BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
 
The following program implements an optimization algorithm line by line using the Particle Optimization (PSO) algorithm. This optimization algorithm works as follows:

1. At first, the environment screen is cleared (clc) all variables are cleared (clear) and all figures are closed (close all).

2. In the problem definition section, we specify a function called CostFunction that measures the performance of the algorithm based on Ans.

3. We define a series of PSO parameters, such as the number of iterations, particle population size, speed range, etc.

4. First, we randomly generate the population of particles and then improve the optimal solution.

5. The improvement of the optimal solution is that for each particle, the speed of the particle is updated based on the previous speed, the best personal position of the particle, and the best collective position of the particles. Then the position of the particle that caused the lower cost is updated. Also, the overall best state (the best collective position of the particles) is updated.

6. This process is repeated until we reach the maximum number of iterations.

7. Finally, a graph of the improvement of the best collective cost of particles in each iteration is displayed.
The program performs the following operations line by line:

1. Clears the command window
2. Clears all the variables in the workspace
3. Closes all open figures or charts
4. Defines a global NFE variable
5. The cost function defines the Sphere
6. nVar defines the number of decision variables
7. VarSize defines the size of the matrix of decision variables
8. It defines the lower and upper bounds of the variables VarMin and VarMax
9. MaxIt sets the maximum number of iterations
10. Set the population size (swarm size), nPop
11. Defines the contraction coefficients, phi1, and phi2, and calculates phi and chi based on these coefficients.
12. Defines VelMax and VelMin speed limits
13. Initializes an empty particle structure
14. Initializes the population of particles with random positions, velocities, costs, and individual best positions and costs.
15. Updates the global best cost based on the particle's personal best cost
16. Initializes arrays to store the best cost and number of performance evaluations for each iteration.
17. Executes the main loop for the specified number of iterations
18. Updates the speed, position, and personal best of each particle based on the current world best
19. Applies velocity and position constraints to each particle
20. It evaluates the position cost of each particle
21. Updates personal and world bests if a better cost is found
22. Stores the best cost and number of performance evaluations for the current iteration
23. Displays the number of iterations, the number of performance evaluations, and the best cost
24. The inertia weight updates w for the next iteration
25. Plots the best cost as a function of the number of performance evaluations
function z=Sphere(x)

    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;

    z=sum(x.^2);

end


This program defines a function called "Sphere" that takes the input vector "x" and returns the sum of the squares of its elements.

- line 'global NFE;' Declare the variable "NFE" as a global variable. This means that it can be accessed and modified from outside the function.

- The "if is empty (NFE)" line checks whether the "NFE" variable is empty. If there is, it means that it has not been initialized yet. In this case, the next line "NFE=0;" Sets the value of "NFE" to 0.

- line "NFE=NFE+1;" Increases the value of "NFE" by 1. This is used to keep track of the number of function evaluations.

- The line `z=sum(x.^2);` calculates the sum of the squares of the elements of the input vector "x" and assigns it to the variable "z".

The purpose of the global variable "NFE" is to keep track of the number of times the function "Sphere" has been evaluated. It is initialized to 0 if it has not yet been assigned a value and incremented by 1 each time the function is called.


The work of a person named Kennedy:
Consider two numbers that are both greater than 0 and two positive numbers whose sum is greater than 4.
The amount of chi = kai = kh is equal to:
Control + r to comment on several lines in the MATLAB program.
We have a separate random number for each component to increase diversity
When we update the position, the position may go out of the space we set for it, so we can consider these restrictions for these positions as well.
Mirror or reflective effect: suppose you have two variables in a 2D space and this particle, our right answer is a black dot, but it finds the fake answer and we have to fix it. We will make a correction process and then it will be approved.
We had a problem in the horizontal direction so this problem does not occur in unlikely movements, we changed its horizontal component and compared it.
That is, those components outside our justified / conditional space, we compare the velocities in those directions. This causes a mirror effect.
We only have to find the places where the variables are out of that range, that is, it is either lower than Varmin or higher than Varmax.
First, we make an image and then we correct it.
Generate a series of random integers: randi() with this command. The first parameter is the interval and we give the desired number.

Solving a PSO problem is generally useful for continuous problems, it is a well-known problem. The problem with energy management is that we have several energy sources to generate electricity and we want to choose between them. Do you know how much energy consumers need for a short period?
Fossil resources, renewable energy sources, biogas from waste and plants for fuel production,
A power plant cannot be shut down, in the lowest working state, it gives us a minimum of power.
It is acceptable to give us any amount between maximum and minimum power
In the case of buying energy from power plants, it works the opposite way. In this problem, we have n power plants, and the decision variable is Pi: the generating power of i, with the following condition:
The cost of using the iam center is:
The cost of pi gradually decreases.
Since pi=0, we have a fixed cost to build that power plant:
We can derive a better model, a ladder diagram for electricity bills. it's like this:
The total cost of consumption is the following figure:
Because the purchase of these power plants is massive, the cost comes down, and the quad function can be used here. In MATLAB.
In the end, it will be our optimization cost
The load that the network asks us is different at different hours of the day and night, let's assume for a moment that we want to calculate that we have:
Production constraint or production capacity
This is a constrained optimization problem that we want to know how to solve.
We implement it in MATLAB.
Production of a series of low limits: min
functionmodel=CreateModel()
 
    pmin=[511 804 501 747 514 655 555 705];
    pmax=[1516 1924 1765 1611 1981 1792 2174 1965];
    
    a0=[8736 5849 8343 9123 6340 6833 5951 6382];
    a1=[8 7 8 6 6 9 6 9];
    a2=[-0.1573 -0.1430 -0.2574 -0.1128 -0.1344 -0.2375 -0.1631 -0.1784]*1e-4;
    
    N=numel(pmin);
    
    PL=10000;
    
    model.N=N;
    model.pmin=pmin;
    model.pmax=pmax;
    model.a0=a0;
    model.a1=a1;
    model.a2=a2;
    model.PL=PL;
 
end
This program defines a function called "CreateModel" that creates a model with certain parameters.

The program first defines the arrays "pmin" and "pmax" which contain the minimum and maximum values for the 8 variables.

Next, the program defines the arrays "a0", "a1" and "a2" that contain the coefficients for a mathematical model.

Then the variable "N" is set to the number of elements of the array "pmin".

The variable "PL" is set to a value of 10000.

Finally, the function sets the fields of the "model" structure with the variables we defined.

The function then terminates and returns the "model" structure.

functionp=CreateRandomSolution(model)
 
    pmin=model.pmin;
    pmax=model.pmax;
    
    p=unifrnd(pmin,pmax);
 
end
This program defines a function called "CreateRandomSolution" that takes a "model" as input. Inside the function, it retrieves the minimum and maximum values for the solution stored in "pain" and "pmax", respectively, from the model.

It then uses the "unifrnd" function to generate a random solution called "p" that lies within the range defined by "pmin" and "pmax".

Finally, the function returns the random solution "p".
function [z sol]=MyCost(p,model)
 
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;
 
    a0=model.a0;
    a1=model.a1;
    a2=model.a2;
    PL=model.PL;
    
    %c=zeros(size(p));
    %for i=1:N
    %    c(i)=a0(i)+a1(i)*p(i)+a2(i)*p(i)^2;
    %end
    
    % Vectorized version of the previous loop
    c=a0+a1.*p+a2.*p.^2;
    
    v=abs(sum(p)/PL-1);
    
    beta=2;
    
    z=sum(c)*(1+beta*v);
    
    sol. p=p;
    sol.pTotal=sum(p);
    sol.c=c;
    sol.cTotal=sum(c);
    sol.v=v;
    sol.z=z;
 
end

function [z sol]=MyCost(p,model)
 
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;
 
    a0=model.a0;
    a1=model.a1;
    a2=model.a2;
    PL=model.PL;
    
    %c=zeros(size(p));
    %for i=1:N
    %    c(i)=a0(i)+a1(i)*p(i)+a2(i)*p(i)^2;
    %end
    
    % Vectorized version of the previous loop
    c=a0+a1.*p+a2.*p.^2;
    
    v=abs(sum(p)/PL-1);
    
    beta=2;
    
    z=sum(c)*(1+beta*v);
    
    sol. p=p;
    sol.pTotal=sum(p);
    sol.c=c;
    sol.cTotal=sum(c);
    sol.v=v;
    sol.z=z;
 
end

The given program defines a function called MyCost that takes two input arguments: p and model. The function calculates and returns the value of a cost function as well as a solution object.

The program starts by initializing a global variable called NFE (number of performance evaluations) to 0 if it is empty.

Then NFE increases by 1.

Then, the program extracts the values of a0, a1, a2, and PL from the model structure.

This program calculates the cost values for each element in p using the coefficients a0, a1, and a2. This is done by elemental multiplication and squaring p and adding the corresponding elements a0, a1, and a2. It can be done using a loop or using vector operations. In the given code, a copy is used to calculate the cost values.

The program also calculates the value of v, which is the absolute difference between the sum of p divided by PL and 1.

A beta variable is set to 2.

The marginal cost, z, is calculated as the sum of the cost values multiplied by the factor (1 + beta * v).

The solution object, sol, is defined and its fields are assigned values. These fields are p (input value p), pTotal (sum of all elements in p), c (calculated cost values), cTotal (sum of all cost values), v (calculated value of v), and z (final cost value ).

Then the function returns z and sol.

CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
CostFunction=@(p) MyCost(p,model);        % Cost Function
 
nVar=model.N;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=model.pmin;         % Lower Bound of Variables
VarMax=model.pmax;         % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=1000;      % Maximum Number of Iterations
 
nPop=100;        % Population Size (Swarm Size)
 
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=2;           % Personal Learning Coefficient
c2=2;           % Global Learning Coefficient
 
% % Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient
 
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
 
%% Initialization
 
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
particle=repmat(empty_particle,nPop,1);
 
GlobalBest.Cost=inf;
 
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=CreateRandomSolution(model);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
BestCost=zeros(MaxIt,1);
 
nfe=zeros(MaxIt,1);
 
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol] = CostFunction(particle(i).Position);
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    nfe(it)=NFE;
    
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');

This program is a PSO (Particle Swarm Optimization) algorithm used to solve an optimization problem.

The program starts by clearing the command window, clearing all variables in the workspace, and closing all shapes.

The problem definition section sets the NFE global variable to zero and creates a model using the CreateModel() function.

In the next step, PSO parameters including the maximum number of iterations (MaxIt), population size (nPop), inertia weight (w), inertia weight damping ratio (damp), and personal/overall learning coefficients (c1 and c2) are defined.

The speed limits, VarMin, and VarMax are also defined based on the lower and upper bounds of the decision variables of the model.

After setting the parameters, the program initializes the particles by creating an empty particle structure and initializing the position, velocity, cost, and solution for each particle in the population. Personal and global best positions and costs are also updated.

In the main PSO loop, the velocity of each particle is updated based on its personal and global best position using the PSO equation. Speed limits are enforced to ensure that the speed remains within the specified range. Then the position of the particle is updated based on the velocity.

If the position of a particle deviates from the specified limits, the velocity mirror effect is applied by reversing the velocity sign. The position is also limited in scope.

The cost and solution are evaluated for each particle, and if the cost of the particle is less than its personal best cost, the personal best is updated. If the individual best cost is less than the global best cost, the global best cost is updated.

The best cost and NFE (Number of Performance Evaluations) are recorded for each iteration.

Finally, the program plots the best cost against the NFE to show the optimization progress.

In general, this program implements a PSO algorithm to find the optimal solution for an optimization problem defined by the model.


The main thing is to have a responsive network. And usually, it doesn't happen either.
One of the techniques of optimization calculations and numerical calculations is the definition of these constraints and their calculations.
The following condition was supposed to be fulfilled, but it was not, now we will define v:
The adverb is the most important thing for us because the structure of the problem is expressed based on the adverb. Especially when the problem has several conditions
Instead of considering the violation as such, let's consider it in another way, suppose:
In the definition of violations:
Suppose we wanted to minimize z in such a way that a condition holds:
so that its violation is v and we want to turn it into an unconstrained problem called z hat:
We have different methods to define this z hat:

We use its multiplicative version:
So we put the code in MATLAB.
MATLAB functions can have several outputs at the same time
Wherever we have .cost, we also add .sol.

functionmodel=CreateModel()
 
    x=[15 65 8 55 21 32 5 88 38 61 44 51 31 30 0 56 11 65 11 44];
    y=[5 99 65 59 95 50 63 74 65 74 60 22 55 95 73 9 59 64 47 57];
    
    N=numel(x);
    
    D=zeros(N,N);
    for i=1:N-1
        for j=i:N
            D(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
            D(j,i)=D(i,j);
        end
    end
    
    model.N=N;
    model.x=x;
    model.y=y;
    model.D=D;
 
end

function xnew=Mutate(x)
 
    [~, Tour]=sort(x);
    
    M=randi([1 3]);
    
    switch M
        case 1
            NewTour=DoSwap(Tour);
            
        case 2
            NewTour=DoReversion(Tour);
            
        case 3
            NewTour=DoInsertion(Tour);
            
    end
    
    xnew=zeros(size(x));
    
    xnew(NewTour)=x(Tour);
 
end
 
 
functionNewTour=DoSwap(Tour)
 
    n=numel(Tour);
    
    i=randsample(n,2);
    i1=i(1);
    i2=i(2);
    
    NewTour=Tour;
    NewTour([i1 i2])=Tour([i2 i1]);
 
end
 
functionNewTour=DoReversion(Tour)
 
    n=numel(Tour);
    
    i=randsample(n,2);
    i1=min(i);
    i2=max(i);
    
    NewTour=Tour;
    NewTour(i1:i2)=Tour(i2:-1:i1);
 
end
 
functionNewTour=DoInsertion(Tour)
 
    n=numel(Tour);
    
    i=randsample(n,2);
    i1=i(1);
    i2=i(2);
    
    if i1<i2
        
        NewTour=[Tour(1:i1) Tour(i2) Tour(i1+1:i2-1) Tour(i2+1:end)];
        
    else
        
        NewTour=[Tour(1:i2-1) Tour(i2+1:i1) Tour(i2) Tour(i1+1:end)];
        
    end
 
end



function [z sol]=MyCost(x,model)
 
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
 
    NFE=NFE+1;
    
    N=model.N;
    D=model.D;
 
    [~, Tour]=sort(x);
    
    L=0;
    
    for k=1:N
        
        i=Tour(k);
        
        if k<N
            j=Tour(k+1);
        else
            j=Tour(1);
        end
        
        L=L+D(i,j);
        
    end
    
    z=L;
    
    sol.Tour=Tour;
    sol.L=L;
 
end



functionPlotSolution(tour,model)
 
    x=model.x;
    y=model.y;
    
    tour=[tour tour(1)];
    
    plot(x(tour),y(tour),'b-s',...
        'LineWidth',2,...
        'MarkerSize',12,...
        'MarkerFaceColor', 'y');
 
end


CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
CostFunction=@(x) MyCost(x,model);        % Cost Function
 
nVar=model.N;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=0;         % Lower Bound of Variables
VarMax=1;         % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=500;      % Maximum Number of Iterations
 
nPop=100;        % Population Size (Swarm Size)
 
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=0.2;           % Personal Learning Coefficient
c2=0.4;           % Global Learning Coefficient
 
% % Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient
 
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
 
%% Initialization
 
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
particle=repmat(empty_particle,nPop,1);
 
GlobalBest.Cost=inf;
 
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
BestCost=zeros(MaxIt,1);
 
nfe=zeros(MaxIt,1);
 
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol] = CostFunction(particle(i).Position);
        
        NewSol.Position=Mutate(particle(i).Position);
        [NewSol.Cost NewSol.Sol]=CostFunction(NewSol.Position);
        ifNewSol.Cost<=particle(i).Cost
            particle(i).Position=NewSol.Position;
            particle(i).Cost=NewSol.Cost;
            particle(i).Sol=NewSol.Sol;
        end
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    NewSol.Position=Mutate(GlobalBest.Position);
    [NewSol.Cost NewSol.Sol]=CostFunction(NewSol.Position);
    ifNewSol.Cost<=GlobalBest.Cost
        GlobalBest=NewSol;
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    nfe(it)=NFE;
    
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
    figure(1);
    PlotSolution(GlobalBest.Sol.Tour,model);
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');
 
functionmodel=CreateModel()
 
    % Item Values
    v=[2   8  11  18   3   2   2   8  19   7  11  17];
 
    % Item Weights
    w=[26  36  50  35  50  25  18  48  27  30  18  19];
 
    % Item Counts
    M=[7   2   3   7   3   7   8   3   6   1   7   7];
    
    % Max Weight
    W=850;
    
    % Number of Items
    N=numel(v);
    
    % Export Model Data
    model.N=N;
    model.v=v;
    model.w=w;
    model.M=M;
    model.W=W;
 
end

functionx=CreateRandomSolution(model)
 
    M=model.M;
    
    x=unifrnd(0,M);
 
end


function [z sol]=MyCost(x,model)
    
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;
 
    v=model.v;
    w=model.w;
    M=model.M;
    W=model.W;
 
    r=M-x;
    SumVR=sum(v.*r);
    SumVX=sum(v.*x);
    SumWX=sum(w.*x);
    Violation=max(SumWX/W-1,0);
    
%     alpha=1000;
%     z=SumVR+alpha*Violation;
    
    beta=10;
    z=SumVR*(1+beta*Violation);
 
    sol.x=x;
    sol.r=r;
    sol.SumVX=SumVX;
    sol.SumVR=SumVR;
    sol.SumWX=SumWX;
    sol.Violation=Violation;
    sol.z=z;
    sol.IsFeasible=(Violation==0);
 
end


CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
CostFunction=@(x) MyCost(x,model);    % Cost Function
 
nVar=model.N;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=0;           % Lower Bound of Variables
VarMax=model.M;     % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=500;          % Maximum Number of Iterations
 
nPop=200;            % Population Size (Swarm Size)
 
w=1;                % Inertia Weight
wdamp=0.99;         % Inertia Weight Damping Ratio
c1=2;               % Personal Learning Coefficient
c2=2;               % Global Learning Coefficient
 
% % Constriction Coefficient
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;               % Inertia Weight
% wdamp=1;             % Inertia Weight Damping Ratio
% c1=chi*phi1;         % Personal Learning Coefficient
% c2=chi*phi2;         % Global Learning Coefficient
 
alpha=0.1;
VelMax=alpha*(VarMax-VarMin);    % Maximum Velocity
VelMin=-VelMax;                 % Minimum Velocity
 
%% Initialization
 
% Create Empty Particle Structure
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
% Initialize Global Best
GlobalBest.Cost=inf;
 
% Create Particles Matrix
particle=repmat(empty_particle,nPop,1);
 
% Initialization Loop
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=CreateRandomSolution(model);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
% Array to Hold Best Cost Values at Each Iteration
BestCost=zeros(MaxIt,1);
 
% Array to Hold NFEs
nfe=zeros(MaxIt,1);
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            + c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Update Velocity Bounds
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirroring
        OutOfTheRange=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(OutOfTheRange)=-particle(i).Velocity(OutOfTheRange);
        
        % Update Position Bounds
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
            end
            
        end
        
        
    end
    
    % Update Best Cost Ever Found
    BestCost(it)=GlobalBest.Cost;
    
    % Update NFE
    life(it)=NFE;
 
    % Show Iteration Information
    ifGlobalBest.Sol.IsFeasible
        Flag=' *';
    else
        Flag=[', Violation = ' num2str(GlobalBest.Sol.Violation)];
    end
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it)) Flag]);
    
    % Inertia Weight Damping
    w=w*wdamp;
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');functionmodel=CreateModel()
 
    % Item Values
    v=[2   8  11  18   3   2   2   8  19   7  11  17];
 
    % Item Weights
    w=[26  36  50  35  50  25  18  48  27  30  18  19];
 
    % Item Counts
    M=[7   2   3   7   3   7   8   3   6   1   7   7];
    
    % Max Weight
    W=850;
    
    % Number of Items
    N=numel(v);
    
    % Export Model Data
    model.N=N;
    model.v=v;
    model.w=w;
    model.M=M;
    model.W=W;
 
end



functionxhat=CreateRandomSolution(model)
 
    N=model.N;
    
    xhat=rand(1,N);
 
end


function [z sol]=MyCost(xhat,model)
    
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;
 
    x=ParseSolution(xhat,model);
    
    v=model.v;
    w=model.w;
    M=model.M;
    W=model.W;
 
    r=M-x;
    SumVR=sum(v.*r);
    SumVX=sum(v.*x);
    SumWX=sum(w.*x);
    Violation=max(SumWX/W-1,0);
    
%     alpha=1000;
%     z=SumVR+alpha*Violation;
    
    beta=10;
    z=SumVR*(1+beta*Violation);
 
    sol.x=x;
    sol.r=r;
    sol.SumVX=SumVX;
    sol.SumVR=SumVR;
    sol.SumWX=SumWX;
    sol.Violation=Violation;
    sol.z=z;
    sol.IsFeasible=(Violation==0);
 
end


functionx=ParseSolution(xhat,model)
 
    M=model.M;
    
    x=min(floor((M+1).*xhat),M);
 
end


CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
CostFunction=@(x) MyCost(x,model);    % Cost Function
 
nVar=model.N;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=0;           % Lower Bound of Variables
VarMax=1;           % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=200;          % Maximum Number of Iterations
 
nPop=200;            % Population Size (Swarm Size)
 
w=1;                % Inertia Weight
wdamp=0.99;         % Inertia Weight Damping Ratio
c1=2;               % Personal Learning Coefficient
c2=2;               % Global Learning Coefficient
 
% % Constriction Coefficient
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;               % Inertia Weight
% wdamp=1;             % Inertia Weight Damping Ratio
% c1=chi*phi1;         % Personal Learning Coefficient
% c2=chi*phi2;         % Global Learning Coefficient
 
alpha=0.1;
VelMax=alpha*(VarMax-VarMin);    % Maximum Velocity
VelMin=-VelMax;                 % Minimum Velocity
 
%% Initialization
 
% Create Empty Particle Structure
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
% Initialize Global Best
GlobalBest.Cost=inf;
 
% Create Particles Matrix
particle=repmat(empty_particle,nPop,1);
 
% Initialization Loop
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=CreateRandomSolution(model);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
% Array to Hold Best Cost Values at Each Iteration
BestCost=zeros(MaxIt,1);
 
% Array to Hold NFEs
nfe=zeros(MaxIt,1);
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            + c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Update Velocity Bounds
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirroring
        OutOfTheRange=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(OutOfTheRange)=-particle(i).Velocity(OutOfTheRange);
        
        % Update Position Bounds
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
            end
            
        end
        
        
    end
    
    % Update Best Cost Ever Found
    BestCost(it)=GlobalBest.Cost;
    
    % Update NFE
    life(it)=NFE;
 
    % Show Iteration Information
    ifGlobalBest.Sol.IsFeasible
        Flag=' *';
    else
        Flag=[', Violation = ' num2str(GlobalBest.Sol.Violation)];
    end
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it)) Flag]);
    
    % Inertia Weight Damping
    w=w*wdamp;
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');


functionmodel=CreateModel()
 
    xs=0;
    ys=0;
    
    xt=4;
    yt=6;
    
    xc=2;
    yc=4;
    r=2;
    
    n=3;
    
    xmin=-10;
    xmax= 10;
    
    ymin=-10;
    ymax= 10;
    
    model.xs=xs;
    model.ys=ys;
    model.xt=xt;
    model.yt=yt;
    model.xc=xc;
    model.yc=yc;
    model.r=r;
    model.n=n;
    model.xmin=xmin;
    model.xmax=xmax;
    model.ymin=ymin;
    model.ymax=ymax;
    
end




functionsol1=CreateRandomSolution(model)
 
    n=model.n;
    
    xmin=model.xmin;
    xmax=model.xmax;
    
    ymin=model.ymin;
    ymax=model.ymax;
 
    sol1.x=unifrnd(xmin,xmax,1,n);
    sol1.y=unifrnd(ymin,ymax,1,n);
    
end


function [z sol]=MyCost(sol1,model)
 
    global NFE;
    if is empty(NFE)
        NFE=0;
    end
    
    NFE=NFE+1;
 
    sol=ParseSolution(sol1,model);
    
    beta=10;
    z=sol.L*(1+beta*sol.Violation);
 
end


functionsol2=ParseSolution(sol1,model)
 
    x=sol1.x;
    y=sol1.y;
    
    xs=model.xs;
    ys=model.ys;
    xt=model.xt;
    yt=model.yt;
    xc=model.xc;
    yc=model.yc;
    r=model.r;
    
    XS=[xs x xt];
    YS=[ys y yt];
    k=numel(XS);
    TS=linspace(0,1,k);
    
    tt=linspace(0,1,100);
    xx=spline(TS,XS,tt);
    yy=spline(TS,YS,tt);
    
    dx=diff(xx);
    dy=diff(yy);
    
    L=sum(sqrt(dx.^2+dy.^2));
    
    d=sqrt((xx-xc).^2+(yy-yc).^2);
    v=max(1-d/r,0);
    Violation=mean(v);
    
    sol2.TS=TS;
    sol2.XS=XS;
    sol2.YS=YS;
    sol2.tt=tt;
    sol2.xx=xx;
    sol2.yy=yy;
    sol2.dx=dx;
    sol2.dy=dy;
    sol2.L=L;
    sol2.Violation=Violation;
    sol2.IsFeasible=(Violation==0);
    
%     figure;
%     plot(xx,yy);
%     hold on;
%     plot(XS,YS,'ro');
%     xlabel('x');
%     ylabel('y');
%     
%     figure;
%     plot(tt,xx);
%     hold on;
%     plot(TS,XS,'ro');
%     xlabel('t');
%     ylabel('x');
%     
%     figure;
%     plot(tt,yy);
%     hold on;
%     plot(TS,YS,'ro');
%     xlabel('t');
%     ylabel('y');
    
end

function sol2=ParseSolution(sol1,model)
 
    x=sol1.x;
    y=sol1.y;
    
    xs=model.xs;
    ys=model.ys;
    xt=model.xt;
    yt=model.yt;
    xc=model.xc;
    yc=model.yc;
    r=model.r;
    
    XS=[xs x xt];
    YS=[ys y yt];
    k=numel(XS);
    TS=linspace(0,1,k);
    
    tt=linspace(0,1,100);
    xx=spline(TS,XS,tt);
    yy=spline(TS,YS,tt);
    
    dx=diff(xx);
    dy=diff(yy);
    
    L=sum(sqrt(dx.^2+dy.^2));
    
    d=sqrt((xx-xc).^2+(yy-yc).^2);
    v=max(1-d/r,0);
    Violation=mean(v);
    
    sol2.TS=TS;
    sol2.XS=XS;
    sol2.YS=YS;
    sol2.tt=tt;
    sol2.xx=xx;
    sol2.yy=yy;
    sol2.dx=dx;
    sol2.dy=dy;
    sol2.L=L;
    sol2.Violation=Violation;
    sol2.IsFeasible=(Violation==0);
    
%     figure;
%     plot(xx,yy);
%     hold on;
%     plot(XS,YS,'ro');
%     xlabel('x');
%     ylabel('y');
%     
%     figure;
%     plot(tt,xx);
%     hold on;
%     plot(TS,XS,'ro');
%     xlabel('t');
%     ylabel('x');
%     
%     figure;
%     plot(tt,yy);
%     hold on;
%     plot(TS,YS,'ro');
%     xlabel('t');
%     ylabel('y');
    
end


functionPlotSolution(sol,model)
 
    xs=model.xs;
    ys=model.ys;
    xt=model.xt;
    yt=model.yt;
    xc=model.xc;
    yc=model.yc;
    r=model.r;
    
    XS=sol.XS;
    YS=sol.YS;
    xx=sol.xx;
    yy=sol.yy;
    
    theta=linspace(0,2*pi,100);
    fill(xc+r*cos(theta),yc+r*sin(theta),'m');
    hold on;
    plot(xx,yy,'km,'LineWidth',2);
    plot(XS,YS,'ro');
    plot(xs, ys,' bs', 'MarkerSize',12, 'MarkerFaceColor','y');
    plot(xt,yt,'kp', 'MarkerSize',16, 'MarkerFaceColor', 'g');
    hold off;
    grid on;
    axis equal;
 
end



CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
model. n=3;
 
CostFunction=@(x) MyCost(x,model);    % Cost Function
 
nVar=model.n;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin.x=model.xmin;           % Lower Bound of Variables
VarMax.x=model.xmax;           % Upper Bound of Variables
VarMin.y=model.ymin;           % Lower Bound of Variables
VarMax.y=model.ymax;           % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=500;          % Maximum Number of Iterations
 
nPop=200;            % Population Size (Swarm Size)
 
w=1;                % Inertia Weight
wdamp=0.99;         % Inertia Weight Damping Ratio
c1=2;               % Personal Learning Coefficient
c2=2;               % Global Learning Coefficient
 
% % Constriction Coefficient
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;               % Inertia Weight
% wdamp=1;             % Inertia Weight Damping Ratio
% c1=chi*phi1;         % Personal Learning Coefficient
% c2=chi*phi2;         % Global Learning Coefficient
 
alpha=0.1;
VelMax.x=alpha*(VarMax.x-VarMin.x);    % Maximum Velocity
VelMin.x=-VelMax.x;                    % Minimum Velocity
VelMax.y=alpha*(VarMax.y-VarMin.y);    % Maximum Velocity
VelMin.y=-VelMax.y;                    % Minimum Velocity
 
%% Initialization
 
% Create Empty Particle Structure
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
% Initialize Global Best
GlobalBest.Cost=inf;
 
% Create Particles Matrix
particle=repmat(empty_particle,nPop,1);
 
% Initialization Loop
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=CreateRandomSolution(model);
    
    % Initialize Velocity
    particle(i).Velocity.x=zeros(VarSize);
    particle(i).Velocity.y=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
% Array to Hold Best Cost Values at Each Iteration
BestCost=zeros(MaxIt,1);
 
% Array to Hold NFEs
nfe=zeros(MaxIt,1);
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % x Part
        
        % Update Velocity
        particle(i).Velocity.x = w*particle(i).Velocity. x ...
            + c1*rand(VarSize).*(particle(i).Best.Position.x-particle(i).Position.x) ...
            + c2*rand(VarSize).*(GlobalBest.Position.x-particle(i).Position.x);
        
        % Update Velocity Bounds
        particle(i).Velocity.x = max(particle(i).Velocity.x,VelMin.x);
        particle(i).Velocity.x = min(particle(i).Velocity.x,VelMax.x);
        
        % Update Position
        particle(i).Position. x = particle(i).Position. x + particle(i).Velocity.x;
        
        % Velocity Mirroring
        OutOfTheRange=(particle(i).Position.x<VarMin.x | particle(i).Position.x>VarMax.x);
        particle(i).Velocity.x(OutOfTheRange)=-particle(i).Velocity.x(OutOfTheRange);
        
        % Update Position Bounds
        particle(i).Position.x = max(particle(i).Position.x,VarMin.x);
        particle(i).Position.x = min(particle(i).Position.x,VarMax.x);
        
        % y Part
        
        % Update Velocity
        particle(i).Velocity.y = w*particle(i).Velocity. y...
            + c1*rand(VarSize).*(particle(i).Best.Position.y-particle(i).Position.y) ...
            + c2*rand(VarSize).*(GlobalBest.Position.y-particle(i).Position.y);
        
        % Update Velocity Bounds
        particle(i).Velocity.y = max(particle(i).Velocity.y,VelMin.y);
        particle(i).Velocity.y = min(particle(i).Velocity.y,VelMax.y);
        
        % Update Position
        particle(i).Position. y = particle(i).Position. y + particle(i).Velocity. y;
        
        % Velocity Mirroring
        OutOfTheRange=(particle(i).Position.y<VarMin.y | particle(i).Position.y>VarMax.y);
        particle(i).Velocity.y(OutOfTheRange)=-particle(i).Velocity.y(OutOfTheRange);
        
        % Update Position Bounds
        particle(i).Position.y = max(particle(i).Position.y,VarMin.y);
        particle(i).Position.y = min(particle(i).Position.y,VarMax.y);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
            end
            
        end
        
        
    end
    
    % Update Best Cost Ever Found
    BestCost(it)=GlobalBest.Cost;
    
    % Update NFE
    life(it)=NFE;
 
    % Show Iteration Information
    ifGlobalBest.Sol.IsFeasible
        Flag=' *';
    else
        Flag=[', Violation = ' num2str(GlobalBest.Sol.Violation)];
    end
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it)) Flag]);
    
    % Plot Solution
    figure(1);
    PlotSolution(GlobalBest.Sol,model);
    
    % Inertia Weight Damping
    w=w*wdamp;
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');


CLC;
clear;
close all;
 
%% Problem Definition
 
global NFE;
NFE=0;
 
model=CreateModel();
 
CostFunction=@(x) MyCost(x,model);        % Cost Function
 
nVar=model.N;       % Number of Decision Variables
 
VarSize=[1 nVar];   % Size of Decision Variables Matrix
 
VarMin=0;         % Lower Bound of Variables
VarMax=1;         % Upper Bound of Variables
 
 
%% PSO Parameters
 
MaxIt=500;      % Maximum Number of Iterations
 
nPop=100;        % Population Size (Swarm Size)
 
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=0.2;           % Personal Learning Coefficient
c2=0.4;           % Global Learning Coefficient
 
% % Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient
 
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
 
%% Initialization
 
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Sol=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Sol=[];
 
particle=repmat(empty_particle,nPop,1);
 
GlobalBest.Cost=inf;
 
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost particle(i).Sol]=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.Sol=particle(i).Sol;
    
    % Update Global Best
    ifparticle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end
 
BestCost=zeros(MaxIt,1);
 
nfe=zeros(MaxIt,1);
 
 
%% PSO Main Loop
 
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost particle(i).Sol] = CostFunction(particle(i).Position);
        
        NewSol.Position=Mutate(particle(i).Position);
        [NewSol.Cost NewSol.Sol]=CostFunction(NewSol.Position);
        ifNewSol.Cost<=particle(i).Cost
            particle(i).Position=NewSol.Position;
            particle(i).Cost=NewSol.Cost;
            particle(i).Sol=NewSol.Sol;
        end
        
        % Update Personal Best
        ifparticle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.Sol=particle(i).Sol;
            
            % Update Global Best
            ifparticle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    NewSol.Position=Mutate(GlobalBest.Position);
    [NewSol.Cost NewSol.Sol]=CostFunction(NewSol.Position);
    ifNewSol.Cost<=GlobalBest.Cost
        GlobalBest=NewSol;
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    nfe(it)=NFE;
    
    disp(['Iteration ' num2str(it) ': NFE = 'num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
    figure(1);
    PlotSolution(GlobalBest.Sol.Tour,model);
    
end
 
%% Results
 
figure;
plot(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');
