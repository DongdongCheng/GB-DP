function   DPC_KNN_PCA(A,percent,line_target)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pca算法

t1=clock
Y1=A;
line_target = line_target

% [coeff,score,latent,tsquared,explained,mu] = pca(A);
[coeff,score,latent,tsquared,explained] = pca(A);
d=1;
energy=sum(explained(1:d));
while energy<90
    d=d+1;
    energy=sum(explained(1:d));
end
newsample=A*coeff(:,1:d);
A=newsample;
disp('come here')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算密度
[N,dim]=size(A);
% dist=zeros(N,N);
%     for i=1:N
%         for j=1:N
%             for k=1:dim
%             dist(i,j)=dist(i,j)+(A(i,k)-A(j,k))^2;
%             end
%             dist(i,j)=sqrt(dist(i,j));
%         end
%     end
dist=pdist2(A,A);
%对距离进行排序
[sdist,index]=sort(dist,2);%对dist按行进行排序
k=floor(percent*N);
rho=zeros(N,1);
for i=1:N
    for j=2:k+1
        rho(i)=rho(i)+dist(i,index(i,j))^2;
    end
    rho(i)=rho(i)/k;
    rho(i)=exp(-rho(i));
end
% disp('The only input needed is a distance matrix file')
% disp('The format of this file should be: ')
% disp('Column 1: id of element i')
% disp('Column 2: id of element j')
% disp('Column 3: dist(i,j)')
% fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);

% position=round(N*percent/100);
% sda=sort(xx(:,3));
% dc=sda(position);
% 
% fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);
% 
% 
% for i=1:ND
%   rho(i)=0.;
% end
% %
% % Gaussian kernel
% %
% for i=1:ND-1
%   for j=i+1:ND
%      rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%      rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%   end
% end
%
% "Cut off" kernel
%
% for i=1:ND-1
%   for j=i+1:ND
%     if (dist(i,j)<dc)
%       rho(i)=rho(i)+1.;
%       rho(j)=rho(j)+1.;
%    end
%  end
% end
ND=N;
maxd=max(max(dist));

[rho_sorted,ordrho]=sort(rho,'descend');
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));
nneigh(ordrho(1))=ordrho(1);
disp('Generated file:DECISION GRAPH')
disp('column 1:Density')
disp('column 2:Delta')

% tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('\rho')
% ylabel ('\delta')
% rect = getrect(1);
% rhomin=rect(1);
% deltamin=rect(2);
% NCLUST=0;
% for i=1:ND
%   cl(i)=-1;
% end
% for i=1:ND
%   if ( (rho(i)>rhomin) && (delta(i)>deltamin))
%      NCLUST=NCLUST+1;
%      cl(i)=NCLUST;
%      icl(NCLUST)=i;
%   end
% end
% hold on;
% cmap=colormap;
% for i=1:NCLUST
%    ic=int8((i*64.)/(NCLUST*1.));
%    hold on
%    plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
% end


fid = fopen('DECISION_GRAPH', 'w');
for i=1:ND
   fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
end

t2 = clock
disp('Select a rectangle enclosing cluster centers')
scrsz = get(0,'ScreenSize');

figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);
%figure(1);
for i=1:ND
  ind(i)=i;
  gamma(i)=rho(i)*delta(i);
end
subplot(2,1,1)


tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
title ('DP Decision Graph','FontSize',15.0)
xlabel ('\rho')
ylabel ('\delta')
subplot(2,1,1)
rect = getrect(1);

t3 =clock
rhomin=rect(1);
deltamin=rect(2);
NCLUST=0;
cl=zeros(ND,1);
for i=1:ND
  cl(i)=-1;
end
for i=1:ND
  if ( (rho(i)>rhomin) && (delta(i)>deltamin))
     NCLUST=NCLUST+1;
     cl(i)=NCLUST;
     icl(NCLUST)=i;
  end
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
disp('Performing assignation')

%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end
%halo
for i=1:ND
  halo(i)=cl(i);
end

t4 = clock

fprintf('running time1: %i \n', etime(t4,t1));
fprintf('running time2: %i \n', etime(t3,t2));
fprintf('running time3: %i \n', etime(t4,t1) - etime(t3,t2));

cmap=colormap;
for i=1:NCLUST
   ic=int8((i*64.)/(NCLUST*1.));
   subplot(2,1,1)
   hold on
   plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end
subplot(2,1,2)
disp('Performing 2D nonclassical multidimensional scaling')
% Y1 = mdscale(dist, 2, 'criterion','metricstress');

plot(Y1(:,1),Y1(:,2),'o','MarkerSize',2,'MarkerFaceColor','k','MarkerEdgeColor','k');
title ('DP Clustering Result','FontSize',15.0)
xlabel ('X')
ylabel ('Y')
for i=1:ND
 A(i,1)=0.;
 A(i,2)=0.;
end
for i=1:NCLUST
  nn=0;
  ic=int8((i*64.)/(NCLUST*1.));
  for j=1:ND
    if (halo(j)==i)
      nn=nn+1;
      A(nn,1)=Y1(j,1);
      A(nn,2)=Y1(j,2);
    end
  end
  hold on
  plot(A(1:nn,1),A(1:nn,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end



% plot(Y1(icl,1),Y1(icl,2),'k*','MarkerSize',10);
%for i=1:ND
%   if (halo(i)>0)
%      ic=int8((halo(i)*64.)/(NCLUST*1.));
%      hold on
%      plot(Y1(i,1),Y1(i,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   end
%end
% faa = fopen('CLUSTER_ASSIGNATION', 'w');
% disp('Generated file:CLUSTER_ASSIGNATION')
% disp('column 1:element id')
% disp('column 2:cluster assignation without halo control')
% disp('column 3:cluster assignation with halo control')
% for i=1:ND
%    fprintf(faa, '%i %i %i\n',i,cl(i),halo(i));
% end
% %画出一个点与其最近邻nneigh
% for i=1:ND
%     plot([Y1(i,1),Y1(nneigh(i),1)],[Y1(i,2),Y1(nneigh(i),2)]);
% end
figure;drawcluster2(Y1,cl,NCLUST+1);
set(gca,'position',[0,0,0.8,0.8]);
% saveas(gcf,'DPC_KNN_6.jpg')
% [new_label] = label_map( cl, line_target );
% nmi=computeNMI(new_label,line_target);
% acc = accuracy(line_target,new_label);
% disp([acc,nmi])

