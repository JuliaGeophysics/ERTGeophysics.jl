# ERTGeophysics package module entry.
# Author: @pankajkmishra
# DC resistivity (ERT): 2.5D (profiles) and 3D forward modelling and Gauss–Newton inversion on rectangular meshes
# with ModEM model files as in MTGeophysics.jl, flat ground or staircase topography, and ModEM-style data files
# carrying the electrodes in WGS84 and local coordinates. 2D: y along the profile; 3D: x north, y east; z DOWN.
module ERTGeophysics

using CairoMakie
using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
import Proj

#----- Core utilities ------------------------------------------------------#
include("CoreUtils.jl")
#----- Mesh, model files, data files, topography and masks ----------------#
include("Mesh2D.jl")
include("Data2D.jl")
include("Topo2D.jl")
include("Mask2D.jl")
include("Control2D.jl")
include("Errors2D.jl")
#----- Forward modelling --------------------------------------------------#
include("Fwd2D.jl")
#----- Inversion ----------------------------------------------------------#
include("Inv2D.jl")
include("Inv2D_GN.jl")
include("Inv2D_NLCG.jl")
#----- Plots ---------------------------------------------------------------#
include("PlotModel2D.jl")
include("PlotData2D.jl")
#----- Mesh design ---------------------------------------------------------#
include("MakeMesh2D.jl")
#----- 3D --------------------------------------------------------------------#
include("Data3D.jl")
include("Mesh3D.jl")
include("Control3D.jl")
include("Fwd3D.jl")
include("Inv3D.jl")
include("PlotModel3D.jl")
include("MakeMesh3D.jl")

#----- Exports: mesh and model files ---------------------------------------#
export ERT2DMesh, ModelFile2D
export BuildMesh2D, RefineMesh2D, Mesh2DFromInputs, ReadModel2D, WriteModel2D
export ert2d_surface, ert2d_is_flat, ert2d_ground_depth, ert2d_air_mask, ert2d_topo_air
export ert2d_electrode_columns, ert2d_electrode_depths, ert2d_electrode_offsets
export ert2d_y_centers, ert2d_z_centers, ert2d_geometric_layers, ert2d_padding, ert2d_padding_cells, ert2d_block_model
export ERT2D_AIR_TAG, ERT2D_AIR_THRESHOLD, ERT2D_MASK_WATER, ERT2D_MASK_PADDING

#----- Exports: data files -------------------------------------------------#
export DataFile2D, FitSummary2D, ndata, nelectrodes
export load_data2d, write_data2d, ImportData2D, BuildSurvey2D, select_data2d, recentre_data2d
export ert2d_local_frame, ert2d_latlon, ert2d_fit_profile
export ert2d_dipole_dipole, ert2d_wenner, ert2d_schlumberger, ert2d_pole_dipole
export ert2d_geometric_factor, ert2d_numerical_geometric_factor, ert2d_pseudo, ert2d_spacing
export ert2d_valid_data, chi2_rms2d
export Reciprocals2D

#----- Exports: topography and masks ---------------------------------------#
export Topo2D, ReadTopo2D, WriteTopo2D, Topography2D, ert2d_profile_topography
export Mask2D, ert2d_ground

#----- Exports: control files -----------------------------------------------#
export FwdCtrl2D, InvCtrl2D, Cov2D
export ReadFwdCtrl2D, WriteFwdCtrl2D, ReadInvCtrl2D, WriteInvCtrl2D
export ReadCov2D, WriteCov2D

#----- Exports: forward -----------------------------------------------------#
export ERT2DForward, ERT2DCache, ert2d_forward, ert2d_resistance, ert2d_homogeneous, ert2d_wavenumbers, ert2d_forward_mesh
export ert2d_sensitivity, ert2d_sensitivity_transpose
export ForwardSolve2D, FrechetDerivative2D, WriteFrechet2D

#----- Exports: inversion ---------------------------------------------------#
export Invert2D, Inv2DOptions, Inv2DResult, AbstractInversion2D
export inv2d_frechet, inv2d_gradient, ert2d_coverage
export inv2d_tag, inv2d_init, inv2d_prepare!, inv2d_direction, inv2d_reject!, inv2d_accept!, inv2d_restart!
export inv2d_info, inv2d_validate, inv2d_cool_always
export GaussNewton2D, GaussNewton2DConfig, GaussNewton2DResult
export NLCG2D, NLCG2DConfig, NLCG2DResult

#----- Exports: plots and mesh design ---------------------------------------#
export plot_ert2d_model, plot_ert2d_mesh, plot_ert2d_pseudosection, plot_ert2d_data_fit
export plot_inv2d_convergence
export PlotModel2D, PlotData2D, PlotInversion2D
export MakeMesh2D

#----- Exports: 3D ------------------------------------------------------------#
export DataFile3D, load_data3d, write_data3d, BuildSurvey3D, select_data3d, recentre_data3d, chi2_rms3d
export ert3d_local_frame, ert3d_latlon, ert3d_grid, ert3d_line_quadrupoles, ert3d_geometric_factor, ert3d_pseudo
export ert3d_spacing
export ERT3DMesh, ModelFile3D, BuildMesh3D, RefineMesh3D, Mesh3DFromInputs, ReadModel3D, WriteModel3D, Topography3D
export ert3d_surface, ert3d_is_flat, ert3d_ground_depth, ert3d_air_mask, ert3d_topo_air, ert3d_padding_mask
export ert3d_forward_mesh
export ert3d_electrode_columns, ert3d_electrode_depths, ert3d_electrode_offsets
export ert3d_x_centers, ert3d_y_centers, ert3d_z_centers, ert3d_block_model
export FwdCtrl3D, InvCtrl3D, ReadFwdCtrl3D, WriteFwdCtrl3D, ReadInvCtrl3D, WriteInvCtrl3D
export ERT3DForward, ERT3DCache, ert3d_forward, ert3d_resistance, ert3d_homogeneous, ert3d_numerical_geometric_factor
export ert3d_sensitivity
export ForwardSolve3D, FrechetDerivative3D
export Invert3D, Inv3DOptions, ert3d_coverage
export plot_ert3d_slices, plot_ert3d_section, plot_ert3d_mesh, plot_ert3d_data_fit, PlotInversion3D
export MakeMesh3D

end # module
