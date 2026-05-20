module MyGeoStatsModule

    ## 0. Loading packages
    using GeoStats
    #using GeoStatsBase
    using GeoStatsProcesses
    using Random
    using Meshes
    using LinearAlgebra


    #export generate_realization_2d, generate_realization_3d

    # Function for 2D field
    function generate_realization_2d(nx, dx, nz, dz, a_major, c_minor, θ, mean=0, std=1)
        # Inputs:
        # nx, dx, nz, dz: number of cells and their dimension in two dimension
        # a_major, a_minor: length of ellipsoid semi-axes
        # θ: rotation angle
        # mean, std: mean and std of field value
        #
        # Output: 2D Gaussian field flattened in a 1D array

        rng = MersenneTwister()
    
        # 1. Create grid
        grid = CartesianGrid((0, 0), (nx*dx, nz*dz), dims=(nx, nz))
    
        # 2. Create variogram
        γ = SphericalVariogram(ranges=(a_major, c_minor), sill=std^2, rotation=Angle2d(θ))

        # 3. Create Gaussian process, generates a standardized field
        realization = rand(rng, GaussianProcess(γ, mean), grid, method=FFTSIM())
    
        # 4. Convert to vector
        realization_vec = realization.field
    
        return realization_vec
    end    # End function 

    # Function for 3D field
    function generate_realization_3d(nx, dx, ny, dy, nz, dz, a_major, b_mid, c_minor, yaw, pitch, roll, mean=0, std=1)
        # Inputs:
        # nx, dx, ny, dy, nz, dz: number of cells and their dimension in three dimension
        # a_major, a_mid, a_minor: length of ellipsoid semi-axes
        # yaw: rotation angle in radius about z axis in up-down 
        # pitch: rotation angle in radius about y axis in west-east
        # roll: rotation angle in radius about x axis in north-south
        # mean, std: mean and std of field value
        #
        # Output: 3D Gaussian field flattened in a 1D array

        rng = MersenneTwister()
    
        # 1. Create grid
        grid = CartesianGrid((0, 0, 0), (nx*dx, ny*dy, nz*dz), dims=(nx, ny, nz))
    
        # 2. Create variogram
        γ = SphericalVariogram(ranges=(a_major, b_mid, c_minor), sill=std^2, rotation=RotZYX(yaw, pitch, roll))

        # 3. Create Gaussian process, generates a standardized field
        realization = rand(rng, GaussianProcess(γ, mean), grid, method=FFTSIM())
    
        # 4. Convert to vector
        realization_vec = realization.field

    
        return realization_vec
    end   # End function

end    # End module

# Example
#real_1 = generate_realization_2d(nx=50, dx=40, nz=50, dz=20, a_major=400, c_minor=200, θ=3π/4)
