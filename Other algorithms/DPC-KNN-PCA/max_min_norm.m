function [D1] = max_min_norm( D )
%��С���淶�����ݼ�
[n,d]=size(D);
disp('----------')
disp(D)
M=max(D);

m=min(D);
D1=zeros(n,d);
for i=1:d
    D1(:,i)=(D(:,i)-m(i))/(M(i)-m(i));
end
end

