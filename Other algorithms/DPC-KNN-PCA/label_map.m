function [ new_label ] = label_map( label, gnd )
%为匹配真实标签，对标签重新调整
K = length(unique(gnd));
cost_mat = zeros(K,K);
for i=1:K
    idx = find(label==i);
    for j=1:K       
        cost_mat(i,j) = length(find(gnd(idx)~=j));
    end
end
[assignment,cost] = munkres(cost_mat);
[assignedrows,dum]=find(assignment');
new_label = label;
for i=1:K
    idx = find(label==i);
    new_label(idx) = assignedrows(i);
end