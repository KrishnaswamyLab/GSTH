using Distributed
addprocs(20)

using PyPlot
using SharedArrays

@everywhere begin

    using NPZ
    using Glob
    using Eirene
    using Printf

    function rescale_coords(coord)
        
        return (coord .- minimum(coord))./(maximum(coord)-minimum(coord))
        
    end;

    fpaths_1 = Glob.glob("control/*.npy")
    fpaths_2 = Glob.glob("Cx43_cKO/*.npy")

    fpaths = vcat(fpaths_1, fpaths_2)
    
    #fpaths = Glob.glob("Homeostatic/*.npy")

    npzwrite("wasserstein_control_cKO_filepaths.npy", fpaths)

    num_files = length(fpaths)
end

wass_H0 = SharedArray{Float64, 2}((num_files, num_files))
wass_H1 = SharedArray{Float64, 2}((num_files, num_files))
wass_H0p1 = SharedArray{Float64, 2}((num_files, num_files))

@sync @distributed for ind_i in 1:num_files
    
    ind_j = 1
    
    wass_H0[ind_i,ind_i] = 0
    wass_H1[ind_i,ind_i] = 0
    wass_H0p1[ind_i,ind_i] = 0
    
    pc1 = rescale_coords(npzread(fpaths[ind_i]))
    PH1 = eirene(transpose(pc1[1,:,:]), model="pc", maxdim=1)
    PD1_H0 = barcode(PH1, dim=0);
    PD1_H1 = barcode(PH1, dim=1);
    PD1 = vcat(PD1_H0[1:end-1,:], PD1_H1)
    
    while ind_j < ind_i 
        
        pc2 = rescale_coords(npzread(fpaths[ind_j]))        
        PH2 = eirene(transpose(pc2[1,:,:]), model="pc", maxdim=1)
        PD2_H0 = barcode(PH2, dim=0);
        PD2_H1 = barcode(PH2, dim=1);
        PD2 = vcat(PD2_H0[1:end-1,:], PD2_H1)
        
        W_H0 = wasserstein_distance(PD1_H0, PD2_H0, p=2, q=2)
        W_H1 = wasserstein_distance(PD1_H1, PD2_H1, p=2, q=2)
        W_H0p1 = wasserstein_distance(PD1, PD2, p=2, q=2)

        wass_H0[ind_j, ind_i] = W_H0
        wass_H0[ind_i, ind_j] = W_H0
        
        wass_H1[ind_j, ind_i] = W_H1
        wass_H1[ind_i, ind_j] = W_H1
        
        wass_H0p1[ind_j, ind_i] = W_H0p1
        wass_H0p1[ind_i, ind_j] = W_H0p1
        
        ind_j += 1
        
    end

    @printf("(%d / %d) completed.", ind_i, num_files);
    
end

npzwrite("wasserstein_control_cKO_H0.npy", wass_H0)
npzwrite("wasserstein_control_cKO_H1.npy", wass_H1)
npzwrite("wasserstein_control_cKO_H0p1.npy", wass_H0p1)