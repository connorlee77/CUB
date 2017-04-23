M = csvread('encoderValues.csv', 1);

M(:,1) = M(:,1);
M(:,2) = M(:,2); 

M = M*2*pi / 2578.33;

dM = diff(M);
rho = 0.035;
W = 0.23/2;

position = zeros(3, length(dM));

for i=2:length(dM)
    
    dPhi1 = dM(i-1,2);
    dPhi2 = dM(i-1,1);
    theta = position(3,i-1);
    position(:,i) = position(:,i-1) + rho/2 * [(dPhi2 - dPhi1)*cos(theta); 
        (dPhi2 - dPhi1)*sin(theta);
        -(dPhi2 + dPhi1)/W];
end
figure
plot(position(1,:), position(2,:))
xlabel('x')
ylabel('y')

figure
plot(position(3,:))
xlabel('time')
ylabel('Orientation (rad)')
