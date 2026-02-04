function CM = colormapJetJBBB(npts)
% creates colormap jet with the yellow bnot too bright.
%copied from original for easier file transfer
CM = colormap(jet(npts));
CM(:,2)=CM(:,2)*0.8;

end

% %plot as
% for n = 1:length(fcii)
%         plot(fcii, Fcii,'o','Color',CM(n,:),'MarkerFaceColor',CM(n,:), 'markerSize',14);
%     end