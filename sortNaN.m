%%Replacing all NaN values with mean value of this particular attributes
function M = sortNaN(V)
V = V';
sum = 0;
count =0;
for i = 1:length(V)
    if (isfinite(V(i)) == 1)
        sum = sum + V(i);
        count = count + 1;
    end
end

notNaN = ~isnan(V);
V(~notNaN) = sum/count;
M = mean(V,1);
end