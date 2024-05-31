function out_paths = action_nn_denoiser(net, in_paths, out_dir)

    % make new directory or skip if directory has been made
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    out_paths = in_paths;

    % visit every image in the batch
    parfor i=1:length(in_paths)
        % construct image save path
        img_path = in_paths(i);
        [~,name,ext] = fileparts(img_path{1});
        save_path = [out_dir, name, ext];

        % only process and save image if this hasn't been done before
        % if ~exist(save_path, 'file')
        shifted = imread(img_path{1});

        if size(shifted,3) == 3
            [r,g,b] = imsplit(shifted);
            recovered_r = denoiseImage(r,net);
            recovered_g = denoiseImage(g,net);
            recovered_b = denoiseImage(b,net);
            recovered = cat(3,recovered_r,recovered_g,recovered_b);
        else
            recovered = denoiseImage(shifted,net);
        end

        imwrite(recovered, save_path);
        % end

        % update out_paths to be returned
        out_paths(i) = {save_path};
    end

end