function out = kuramoto_runner()

    kappa = 0.4;
    beta = 0.3;
    
    out = kuramoto(64, kappa, beta);
    save("kuramoto_test.mat", 'out')

end