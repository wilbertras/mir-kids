function MakeGoodFigureBB(dx,dy, FT, fn, saveonlypng)
%makes current figure looking good
%identical to MakeGoodFigure.m, but copied here for easy ues
%dx = x size in incgh
%dy = y size in incgh
%FT is font size (optional)
%fn is filename (array of char, 'example'. If added, file is saved in png
%and fig
if nargin == 0
    dx = 8;
    dy = 5;
    FT = 15;
    fn = [];
    saveonlypng = 0;
elseif nargin == 2
    FT = 15;
    fn = [];
    saveonlypng = 0;
elseif nargin == 3
    fn = [];
    saveonlypng = 0;
elseif nargin == 4
    saveonlypng = 0;
end


set(gcf,'InvertHardcopy','on')
set(gcf,'PaperUnits','inches')
set(gcf,'Units','inches')
myfigsize =  [0.1, 0.1, dx, dy];
set(gcf,'Position', myfigsize);
set(gcf,'PaperPosition',myfigsize)
set(gcf,'PaperSize',[myfigsize(3) myfigsize(4)])
set(gcf,'Color','White')
handles=findobj(gcf,'Type','axes');
for n = 1 : length(handles)
    set(handles(n),'FontSize',FT)
    %ax = gca;
    ax = handles(n);
    ax.YLabel.FontSize = FT+2;ax.XLabel.FontSize = FT+2;
end
set(gcf,'PaperPositionMode','auto');
handles=findobj(gcf,'Type','legend');
for n = 1 : length(handles)
    %h = legend;
    %set(h,'FontSize',FT)
    set(handles(n),'FontSize',FT)
end
%h=get(gcf,'children');
%set(h,'visible','off') SWITCH OF LEGEND
% optional save
if ischar(fn)
    if saveonlypng ~= 1
        saveas(gcf,[fn '.fig'])
    end
    set(gcf,'PaperPositionMode','auto');
    print([fn '.png'],'-dpng','-r300')
end
end