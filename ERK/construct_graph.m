close all; clear all;

load FinalData_For_Jess.mat

experiments = fieldnames(FinalData);

for exp = 1:numel(experiments)

    expname = experiments{exp};
    dat = FinalData.(expname);
    
    xpos = dat.nuclei.x;
    ypos = dat.nuclei.y;
    
    wells = unique(dat.Pos);
    
    for wid = 1:numel(wells)
    
        idx = find(dat.Pos == wells(wid));
        
        cellmean = dat.RatioCNERKNorm(idx,:);
        
        xmean = mean(xpos(idx,:), 2, 'omitnan');
        ymean = mean(ypos(idx,:), 2, 'omitnan');
        
        G = knngraph([xmean ymean], 6);
        A = adjacency(G);
        A = A + A';
        G = graph(A);
        
        pltname = strcat(expname, "_", num2str(wells(wid)), ".png");
        fig = figure();
        p = plot(G, 'XData', xmean, 'YData', ymean, 'NodeLabel', {});
        p.LineWidth = 0.1;
        saveas(gcf, pltname)
        close(fig);
        
        ofname = strcat(expname, "_", num2str(wells(wid)), ".mat");
        save(ofname, "cellmean", "A", "xmean", "ymean")
    
    end
end