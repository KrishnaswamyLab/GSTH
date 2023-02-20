dirData = dir('K10/*.mat');

% load data
matfile = dirData(1);
load(fullfile(matfile.folder, matfile.name))
    
num_timepoints = size(cells_mean, 1);
num_cells = size(cells_mean, 2);

centroids = zeros(num_cells, 2);
cell_types = zeros(num_cells, 1);

% compute centroids
for cid = 1:num_cells

    img = zeros(d1, d2);
    cell_types(cid) = events_info(cid, 8);
    px_list = cell2mat(cells(cid));
    img(px_list) = 1;
    stats = regionprops(img);
    centroids(cid,:) = stats.Centroid;

end

% set signaling threshold
mean_signal = mean(cells_mean, 1);
signal_threshold = prctile(mean_signal, 75);

% plot cell centroid locations colored by cell type and mean signal
plot_tissue(centroids, cell_types, max([d1 d2]), mean_signal)

dr = 5;
rad = 0:dr:max(d1,d2);

% spatial correlation of all cells
rdf_all = zeros(1, numel(rad));
[corrfunc, r, rw] = twopointcorr(centroids(:,1), centroids(:,2), dr, 1000, false);
rdf_all(r./dr) = rdf_all(r./dr) + corrfunc;

% spatial correlation of signaling cells averaged over timepoints

rdf_signaling = zeros(1, numel(rad));
num_obs_signaling = zeros(1, numel(rad));

rdf_signaling_K10pos = zeros(1, numel(rad));
num_obs_signaling_K10pos = zeros(1, numel(rad));

rdf_signaling_K10neg = zeros(1, numel(rad));
num_obs_signaling_K10neg = zeros(1, numel(rad));

for tp = 1:num_timepoints

    is_signalling = cells_mean(tp,:) > signal_threshold;
    ctype_subset = cell_types(is_signalling);
    cell_pos = centroids(is_signalling,:);
    if size(cell_pos,1) > 1
        K10pos_pos = cell_pos(ctype_subset == 1,:);
        K10neg_pos = cell_pos(ctype_subset == 0,:);
        [corrfunc, r, rw] = twopointcorr(cell_pos(:,1), cell_pos(:,2), dr, 1000, false);
        rdf_signaling(r./dr) = rdf_signaling(r./dr) + corrfunc;
        num_obs_signaling(r./dr) = num_obs_signaling(r./dr) + 1;
        if size(K10pos_pos, 1) > 1
            [corrfunc, r, rw] = twopointcorr(K10pos_pos(:,1), K10pos_pos(:,2), dr, 1000, false);
            rdf_signaling_K10pos(r./dr) = rdf_signaling_K10pos(r./dr) + corrfunc;
            num_obs_signaling_K10pos(r./dr) = num_obs_signaling_K10pos(r./dr) + 1;
        end
        if size(K10neg_pos, 1) > 1
            [corrfunc, r, rw] = twopointcorr(K10neg_pos(:,1), K10neg_pos(:,2), dr, 1000, false);
            rdf_signaling_K10neg(r./dr) = rdf_signaling_K10neg(r./dr) + corrfunc;
            num_obs_signaling_K10neg(r./dr) = num_obs_signaling_K10neg(r./dr) + 1;
        end
    end
end

rdf_signaling_norm = rdf_signaling./num_obs_signaling;
rdf_signaling_K10pos = rdf_signaling_K10pos./num_obs_signaling_K10pos;
rdf_signaling_K10neg = rdf_signaling_K10neg./num_obs_signaling_K10neg;

% plot cell density vs. distance
xs = pixels2um(rad);
figure()
plot(xs(2:end), rdf_signaling_norm(2:end), "LineWidth", 1.5)
hold on
plot(xs(2:end), rdf_all(2:end), "LineWidth", 1.5)
plot(xs(2:end), rdf_signaling_K10pos(2:end), "LineWidth", 1.5)
plot(xs(2:end), rdf_signaling_K10neg(2:end), "LineWidth", 1.5)
xlabel("Distance [um]",  'FontSize', 14)
ylabel("Cell Density", 'FontSize', 14)
legend("Signaling", "All Cells", "K10+", "K10-")
xlim([xs(2), 250])

% helper functions

function plot_tissue(pos, labels, ax_lim, signal)

    figure()

    K10pos = scatter(pos(labels==1,1), pos(labels==1,2), 15, "blue", "filled");
    K10pos.AlphaData = signal(labels==1);
    K10pos.MarkerFaceAlpha = "flat";
    
    hold on

    K10neg = scatter(pos(labels==0,1), pos(labels==0,2), 15, "red", "filled");
    K10neg.AlphaData = signal(labels==0);
    K10neg.MarkerFaceAlpha = "flat";

    legend("K10+", "K10-")
    xlim([0, ax_lim])
    ylim([0, ax_lim])
    set(gca,'xcolor','w','ycolor','w','xtick',[],'ytick',[])
    set(gca,'box','off')
    set(gcf,'color','w')
    hold off

end

function micron_distances = pixels2um(pixel_distances)
    micron_distances = (500/1013)*pixel_distances;
end