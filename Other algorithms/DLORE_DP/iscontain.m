function flag=iscontain( q,x )
%�����q��������
[m,n]=size(q);
flag=0;
for i=1:m
    if q(i)==x
        flag=1;
        break;
    end
end

end

