%�����޸ĵ�Ѱ�Һ��ĵ���㷨
function [dist,index,supk,max_nb,rho,local_core,cores,cl,cluster_number] = CoreSearch2(A)
[N,dim]=size(A);
dist=zeros(N,N);
    for i=1:N
        for j=1:N
            for k=1:dim
            dist(i,j)=dist(i,j)+(A(i,k)-A(j,k))^2;
            end
            dist(i,j)=sqrt(dist(i,j));
        end
    end
[sdist,index]=sort(dist,2);%��dist���н�������
%��ʼ����������
r=1;
flag=0;         
nb=zeros(1,N);  %��Ȼ�ھӸ��� 
%NNN=zeros(N,N); %�������Ȼ�ھӼ�
count=0;        %��Ȼ�������Ϊ���������������ͬ�Ĵ���
count1=0;       %ǰһ����Ȼ�������Ϊ���������
count2=0;       %�˴���Ȼ�������Ϊ���������

%������Ȼ����ھ�
while flag==0
    for i=1:N
        k=index(i,r+1);
        nb(k)=nb(k)+1;
      %  NNN(k,nb(k))=i;
    end
    r=r+1;
    count2=0;
    for i=1:N
        if nb(i)==0
            count2=count2+1;
        end
    end
    %����nb(i)=0�ĵ�������������仯�Ĵ���
    if count1==count2
        count=count+1;
    else
        count=1;
    end
    if count2==0 || (r>2 && count>=2)   %�ھ�������ֹ����
        flag=1;
    end
    count1=count2;
end

%������Ȼ����ڵĸ���������
supk=r-1;               %����Kֵ��Ҳ����Ȼ����ھӵ�ƽ����
max_nb=max(nb);         %��Ȼ�ھӵ������Ŀ
min_nb=min(nb);         %��Ȼ�ھӵ���С��Ŀ
%NN=index(:,2:SUPk+1);   %�����ݵ��K�������ݵ㼯
%ratio_nb=nb./(N*SUPk);  %�����ݵ����Ȼ����ھ���Ŀ��ռ����
%����ÿ�����ݵ���ܶ�
%disp(SUPk);
%�������Ӿ���
%disp(supk);
%disp(max_nb);
%disp(min_nb);
rho=zeros(N,1);
Non=max_nb;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%������Ȼ����ͼ
conn=zeros(N,N);
for i=1:N
    for j=2:supk+1
        x=index(i,j);
        conn(i,x)=1/(1+dist(i,x));%����ĵ�����Ϊ��������ƶ�
        conn(x,i)=conn(i,x);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:N
    d=0;
    for j=1:Non+1
        d=d+sdist(i,j);
    end
    rho(i)=(Non/d);
end
[rho_sorted,ordrho]=sort(rho,'descend');%ordrho�����ܶȴӴ�С��˳��
local_core=zeros(N,1);%���n����ľֲ����ĵ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����ܶȴӴ�С��˳��ȥ�������ݵ�׼���ĵ�
for i=1:N
%      if local_core(ordrho(i))==0
         p=ordrho(i);
         maxrho=rho(p);
         maxindex=p;
         for j=1:nb(p)+1%���k�������ܶ����ĵ���Ϊi����k���ڵĺ��ĵ�
             x=index(p,j);
             if maxrho<rho(x)
                 maxrho=rho(x);
                 maxindex=x;
             end
         end
         %�Ծ�������ܶȵĵ����ֲ������
         if local_core(maxindex)==0%���������ܶȵ�Ҳû�з�����ĵ�
             local_core(maxindex)=maxindex;            
         end
         %�õ���p��k���ڵľֲ������
         for j=1:nb(p)+1
             if local_core(index(p,j))==0%�����j�����ڻ�û�д����
                 local_core(index(p,j))=local_core(maxindex);
             else%�����j�������Ѿ��д�����ˣ�ѡ�����Ͻ���һ���������Ϊ�µĴ����
                 q=local_core(index(p,j));
                 if dist(index(p,j),q)>dist(index(p,j),local_core(maxindex))%rho(local_core(maxindex))>=rho(q)%
                     local_core(index(p,j))=local_core(maxindex);
                 end
             end 
             for m=1:N
                 if local_core(m)==index(p,j)
                     local_core(m)=local_core(index(p,j));
                 end
             end
         end
         
%        for j=1:Non+1
%         if local_core(index(ordrho(i),j))==0||local_core(index(ordrho(i),j))~=0&&rho(local_core(index(ordrho(i),j)))<rho(maxindex)%dist(index(ordrho(i),j),local_core(index(ordrho(i),j)))>dist(index(ordrho(i),j),maxindex)%%������㻹û�з�����ĵ�����Ѿ�������ĵ㣬�Ƚ����������ĵ��ĸ��ȽϽ��ͷ��䵽�ĸ����ĵ���
%             local_core(index(ordrho(i),j))=maxindex;
%             delta(index(ordrho(i),j))=dist(index(ordrho(i),j),maxindex);
%             if local_core(maxindex)==0%�������ܶȽϴ�ĵ㻹û�з�����ĵ㣬��Ϊ�������ĵ�Ϊ���Լ�
%                 local_core(maxindex)=maxindex;
%            else%���������Ѿ�������ĵ㣬�Ƚ��������ĵ���ܶȣ�ȡ�ϴ��ܶȵĺ��ĵ���Ϊ���ĵ�
% %                if rho(local_core(index(ordrho(i),j)))<rho(maxindex)
%            local_core(index(ordrho(i),j))=local_core(maxindex);
%            delta(index(ordrho(i),j))=dist(index(ordrho(i),j),local_core(maxindex));
% %                end
%             end
%         end 
%         for m=1:N
%              if local_core(m)==index(ordrho(i),j)
%                  local_core(m)=local_core(index(ordrho(i),j));
%              end
%         end
%          
%       end
%     end    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�õ�׼���ĵ㣬׼���ĵ���������Ϊ���ĵ�
 cluster_number=0;
 cl=zeros(N,1);
for i=1:N
    if local_core(i)==i;
       cluster_number=cluster_number+1;
       cores(cluster_number)=i;
       cl(i)=cluster_number;
    end
end
disp('��ʼ�Ӵظ���Ϊ��');disp(cluster_number);
% �����ǵó�׼����ֱ�ӵõ����Ӵ�
for i=1:N
    cl(i)=cl(local_core(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�������ĵ�ͼ�Լ���Ӧ�ĳ�ʼ������
plot(A(:,1),A(:,2),'.');
hold on;
for i=1:N
    plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)]);
    hold on;
end
%drawcluster2(A,cl,cluster_number+1);
hold on;
plot(A(local_core,1),A(local_core,2),'r.','MarkerSize',8);
% title('����maxnb���ڵĽ��');
end


