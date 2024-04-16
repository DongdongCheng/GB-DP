function [ cl,NCLUST ] = DP(Y1, dist,rho )
%给定数据集A，局部核心点之间的距离，局部核心点的密度，利用DP算法进行聚类
%Y1局部核心点信息
%dist局部核心点之间的距离
%rho局部核心点的密度
maxd=max(max(dist)) * 10 ;
[ND,~]=size(Y1);
[rho_sorted,ordrho]=sort(rho,'descend');
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1 
     if(dist(ordrho(ii),ordrho(jj)) < delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));
nneigh(ordrho(1))=ordrho(1);
maxdelta=max(delta);
delta2=delta;
delta2(find(delta==maxdelta))=0;
secmaxdelta=max(delta2);
delta(find(delta==maxdelta))=2*secmaxdelta;
disp('Generated file:DECISION GRAPH')
disp('column 1:Density')
disp('column 2:Delta')
toc;
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
rect = getrect;


tic;
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
      t = nneigh(ordrho(i));
       if((t==-1))
          t1=1
      else
          t1 = nneigh(ordrho(i));
       end                                  
      cl(ordrho(i))=cl(t1);
  end
end
%halo
for i=1:ND
  halo(i)=cl(i);
end
toc;
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
plot(Y1(icl,1),Y1(icl,2),'k*','MarkerSize',10);
%for i=1:ND
%   if (halo(i)>0)
%      ic=int8((halo(i)*64.)/(NCLUST*1.));
%      hold on
%      plot(Y1(i,1),Y1(i,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   end
%end
% %画出一个点与其最近邻nneigh
% for i=1:ND
%     plot([Y1(i,1),Y1(nneigh(i),1)],[Y1(i,2),Y1(nneigh(i),2)]);
% end
%figure(2);drawcluster2(Y1,cl,NCLUST+1);
end

