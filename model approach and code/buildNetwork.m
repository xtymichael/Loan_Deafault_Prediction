% clear all the NaN and replace it with mean value
n271 = sortNaN(copy271);
n274 = sortNaN(copy274);
n527 = sortNaN(copy527);
n528 = sortNaN(copy528);
nloss = sortNaN(losscopy);

%GRNN - 2 General Regression Neural Network (GRNN)
input = [n271-n274; n528-n527 ];
T = nloss;
net = newgrnn(input, T);

