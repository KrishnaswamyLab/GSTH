dirData = dir('K10/*.mat');

K10_pos = [];
K10_neg = [];

% iterate over all the files
for fid = 1:numel(dirData)

    % load data
    matfile = dirData(fid);
    load(fullfile(matfile.folder, matfile.name))
    
    num_timepoints = size(cells_mean, 1);
    num_cells = size(cells_mean, 2);
    
    % iterate over cells
    for cid = 1:num_cells
    
        cell_signal = cells_mean(:,cid);
        ctype = events_info(cid, 8);
        peak_idxs = find(overallEvents(:,1) == cid);
        num_peaks = numel(peak_idxs);
    
        % Plot for debugging
        % plot_signal(cid, cell_signal, ctype, num_timepoints, peak_idxs, overallEvents)

        if num_peaks > 1
            peak_intervals = zeros((num_peaks-1), 1);
            for peak_idx = 2:num_peaks
                peak_intervals(peak_idx-1) = overallEvents(peak_idxs(peak_idx),2) - overallEvents(peak_idxs(peak_idx-1),2);
            end
            peak_intervals = t2min(peak_intervals);
            if ctype == 1
                K10_pos = [K10_pos; peak_intervals];
            else
                K10_neg = [K10_neg; peak_intervals];
            end
        end
    
    end

end

% plot
[h,L,MX,MED] = violin({K10_pos, K10_neg}, 'xlabel', {'K10+', 'K10-'});
ylabel('Time [min]', 'FontSize', 14)

all_obs = [K10_pos; K10_neg];
labels = [repmat("K10+", numel(K10_pos), 1); repmat("K10-", numel(K10_neg), 1)];
p = kruskalwallis(all_obs, labels);
H = sigstar({[1,2]}, p);
ylabel('Time [min]', 'FontSize', 14)

function plot_signal(cid, sig, ctype, nt, peak_indices, events)
    figure()
    plot(t2min(1:nt), sig)
    hold on
    for peak_idx = 1:numel(peak_indices)
        peak_start = events(peak_indices(peak_idx), 2);
        peak_width = events(peak_indices(peak_idx), 3);
        peak_height = events(peak_indices(peak_idx), 4);
        area(t2min([peak_start, peak_start+peak_width]), [peak_height, peak_height], ...
            'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', 'red');
    end
    hold off
    xlim(t2min([0, nt]))
    ylim([min(sig), max(sig)])
    xlabel('Time (min)')
    if ctype == 1
        title(strcat("Cell ID: ", num2str(cid), " (K10+)"))
    else
        title(strcat("Cell ID: ", num2str(cid), " (K10-)"))
    end
end

function t_scaled = t2min(t)
    t_scaled = (2/60)*t;
end