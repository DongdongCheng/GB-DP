function acc = Acc(Label1,Label2)
%Label1:真实标签 Label2:映射后的标签

T= Label1==Label2;
acc=sum(T)/length(Label2);

end