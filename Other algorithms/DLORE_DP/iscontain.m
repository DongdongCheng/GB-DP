function flag=iscontain( q,x )
%这里的q是列向量
[m,n]=size(q);
flag=0;
for i=1:m
    if q(i)==x
        flag=1;
        break;
    end
end

end

