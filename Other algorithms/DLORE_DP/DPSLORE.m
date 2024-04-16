function [ cl2,sharedCount,short_path] = DPSLORE( A )
%���ھֲ����ĵ㹲����ھ����DP�����㷨
%   ����˼���ǻ�þֲ����ĵ㣬Ȼ�����¶���ֲ����ĵ�֮����ڹ�����ڵľ��룬�����ܶȶ�����µľ��붨�壬����DP�㷨�Ծֲ����ĵ���о��࣬��󽫾ֲ����ĵ�ľ�������չ��ԭʼ���ݼ���
%A ���ݼ�
%stopK������
tic;
%  [A] = max_min_norm(A);
[N,dim]=size(A);
%�����������ĵ�
% [index,supk,max_nb,rho,local_core,cores,cl,cluster_number ] = CoreSearch4(A);
% [index,supk,nb,rho,local_core,cores,cl,cluster_number ] = CoreSearch3(A);
  [dist,index,supk,max_nb,rho,local_core,cores,cl,cluster_number] = CoreSearch2(A);
% [cores,local_core,cl,supk,cluster_number,index] = KeyPoint(A);
%���ú��ĵ�֮��ľ��빹����С������

%��һ�ַ���������ÿ�����������ص�k���ڵĽ���
[rho_sorted,ordrho]=sort(rho,'ascend');
% alpha=0.15;%0.05;
rho_threshold=0;%rho_sorted(floor(N*alpha));%0;
for i=1:cluster_number
    if cores(i)~=0
    if rho(cores(i))<rho_threshold %�ų��ܶȽ�С�ĺ��ĵ�
        mind=inf;
        p=0;
        for j=1:cluster_number
            if i~=j
                x=A(cores(i),:);
                y=A(cores(j),:);
                distance=sqrt(sum((x-y).^2));
%                 distance=pdist2(A(cores(i),:),A(cores(j),:));
            if mind>distance&&rho(cores(j))>rho_threshold
                mind=distance;
                p=j;
            end
            end
        end
        for j=1:N
            if local_core(j)==cores(i)
                local_core(j)=cores(p);
            end
        end
    end
    end
end
cluster_number=0;
cl=zeros(N,1);
for i=1:N
    if local_core(i)==i;
       cluster_number=cluster_number+1;
       cores2(cluster_number)=i;
       cl(i)=cluster_number;
    end
end
% disp('��ʼ�Ӵظ���Ϊ��');disp(cluster_number);
% �����ǵó�׼����ֱ�ӵõ����Ӵ�
for i=1:N
    cl(i)=cl(local_core(i));
end
% %�������ĵ�ͼ�Լ���Ӧ�ĳ�ʼ������
% figure(1);
% plot(A(:,1),A(:,2),'.');
% hold on;
% for i=1:N
%     plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)]);
%     hold on;
% end
% % drawcluster2(A,cl,cluster_number+1);
% hold on;
% plot(A(local_core,1),A(local_core,2),'ro','MarkerSize',5,'MarkerFaceColor','r','MarkerEdgeColor','r');
% hold off;
cdata=cell(1,cluster_number);%����ÿ�����ж�����Щ��
cdataexp=cell(1,cluster_number);%����ÿ�����еĵ㼰ÿ�����е�k����
nc=zeros(1,cluster_number);%��������ĳ�����ĵ�ĵ���
ncexp=zeros(1,cluster_number);
core_dist=zeros(cluster_number,cluster_number);
for i=1:cluster_number
    for j=i+1:cluster_number
        x=A(cores2(i),:);
        y=A(cores2(j),:);
        d=sqrt(sum((x-y).^2));
        core_dist(i,j)=d;
        core_dist(j,i)=d;
    end
end

maxd=max(max(core_dist));
sd=zeros(cluster_number,1);
for i=1:cluster_number
    %����ÿ�����еĵ�
    nc(i)=0;
    ncexp(i)=0;
    x=A(cores2(i),:);
    for j=1:N
        if cl(j)==i
            nc(i)=nc(i)+1;
            y=A(j,:);
            sd(i)=sd(i)+sqrt(sum((x-y).^2));
            ncexp(i)=ncexp(i)+1;
            cdata{1,i}(1,nc(i))=j;
            cdataexp{1,i}(1,ncexp(i))=j;
        end
    end
    %Ѱ�ҵ�i������ÿ�������Non����ھӼ��뵽�ô���
    for j=1:ncexp(i)
       x=cdata{1,i}(1,j);
       d2=sqrt(sum((A(x,:)-A(cores2(i),:)).^2));
%        if d2<1.5*sd(i)/nc(i)
       for k=2:supk+1
           kneighbor=index(x,k);
           if iscontain(cdataexp{1,i}',kneighbor)==0&&rho(kneighbor)>rho_threshold
               ncexp(i)=ncexp(i)+1;
               cdataexp{1,i}(1,ncexp(i))=kneighbor;
           end
       end
%        end
    end 
   
end
%��������������֮������¶���ľ���dist/intersect

sim=zeros(cluster_number,cluster_number);
sharedCount=zeros(cluster_number,cluster_number);
for i=1:cluster_number
    for j=i+1:cluster_number
        inset1=intersect(cdataexp{1,i},cdataexp{1,j});
%         inset2=intersect(cdata{1,i},cdataexp{1,j});
        averho=sum(rho(inset1));
        [~,numinset1]=size(inset1);
        sharedCount(i,j)=numinset1;
        sharedCount(j,i)=numinset1;
%         [~,numinset2]=size(inset2);
%         fprintf('��%d����͵�%d����Ľ�����Ϊ��%d,�ܶȺ�Ϊ%f\n',i,j,numinset1,averho);
        if numinset1==0%&&numinset2==0
            core_dist(i,j)=maxd;%core_dist(i,j);
            core_dist(j,i)=core_dist(i,j);
        else
%             dist1=0;
%             dist2=0;
%             for k=1:numinset1
%                 dist1=dist1+sqrt(sum((A(cores(i),:)-A(inset1(k),:)).^2));
%                 dist2=dist2+sqrt(sum((A(cores(j),:)-A(inset1(k),:)).^2));
%             end
%             core_dist(i,j)=(dist1+dist2)/(numinset1^2);
            core_dist(i,j)=core_dist(i,j)/(averho*numinset1);
            core_dist(j,i)=core_dist(i,j);
        end
        
    end
end
short_path=zeros(cluster_number,cluster_number);
for i=1:cluster_number
     short_path(i,i)=maxd;
     [D,Z]=dijkstra2(core_dist,i);%D�д�ŵ���ԭ�㵽����ÿ����������·���ĳ��ȣ�Z���ŵ���ÿ�������������
     for j=i+1:cluster_number
         short_path(i,j)=D(j);
         if short_path(i,j)==inf
             short_path(i,j)=0;
         end
         
         short_path(j,i)=short_path(i,j);
     end
end
% maxsp=max(max(short_path));
% short_path(find(short_path==maxsp))=2*maxsp;
%����DP�㷨�Ծֲ����ĵ���о���
core_data=A(cores2,:);
core_rho=rho(cores2);
[ core_cl,ncluster ] = DP(core_data,short_path,core_rho );
% [ core_cl,ncluster ] = DP2SNN(core_data, short_path,core_rho,sharedCount,supk/2 );
cl2=zeros(N,1);
cl2(cores2)=core_cl;
for i=1:N
    cl2(i)=cl2(local_core(i));
end

%csvwrite('./data/label2.csv',cl2,0,0);
% SD=sparse(core_dist);
% UG=tril(SD);
% [ST,pred] = graphminspantree(UG,'METHOD','Prim');
% figure(2);
% plot(A(:,1),A(:,2),'.');
% hold on;
% % for i=1:N
% %     if local_core(i)~=0;
% %     plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)],[A(i,3),A(local_core(i),3)]);
% %     hold on;
% %     end
% % end
% % plot(A(cores,1),A(cores,2),'r*','MarkerSize',8);
% % hold on;
figure;
drawcluster2(A,cl2,ncluster);

end

