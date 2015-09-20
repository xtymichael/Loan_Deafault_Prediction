% clear all the NaN and replace it with mean value
nt271 = sortNaN(test271);
nt274 = sortNaN(test274);
nt527 = sortNaN(test527);
nt528 = sortNaN(test528);

%SIMULATION
temp = [nt271 - nt274, nt528 - nt527];

for i = 1:210945
    Predict = temp(:,i);
    newresult(i) = sim(net,Predict);
end