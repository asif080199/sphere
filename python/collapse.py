#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = True
collapse       = True
rendering      = True
plots          = True

# Number of particles
np = 1e4

# Common simulation id
sim_id = "collapse"

### INITIALIZATION ###

# New class
init = sphere.sim(np = np, sid = sim_id + "-init")

# Save radii
init.generateRadii(radius_mean = 0.1)

# Use default params
init.defaultParams(
        k_n = 1.0e6, k_t = 1.0e6,           # normal and tangential stiffness
        gamma_n = 100.0, gamma_t = 100.0,   # normal and tangential viscosity
        mu_s = 0.3, mu_d = 0.3)             # static and dynamic friction

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = numpy.array([hcells, hcells, 1e9]))

# Choose the tangential contact model.
# 1) Visco-fricitonal (fast, incorrect)
# 2) Elastic-viscous-frictional (slow, more correct)
init.contactmodel[0] = 1

# Add gravitational acceleration along the negative z-axis
init.g[2] = -10.0

# Set duration of simulation, automatically determine timestep, etc.
init.initTemporal(total = 10.0)

if (initialization == True):

    # Run sphere
    init.run(dry = True)
    init.run()

    if (plots == True):
        # Make a graph of energies
        init.visualize("energy", savefig=True, outformat='png')

### COLLAPSE ###

# New class
coll = sphere.sim(np = init.np, nw = init.nw, sid = sim_id)

# Read last output file of initialization step
lastf = status(sim_id + "-init")
coll.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf),
        verbose = False)

# Setup collapse experiment by moving the +x boundary and resizing the grid
resizefactor = 3
coll.L[0] = coll.L[0]*resizefactor # world length in x-direction (0)
coll.num[0] = coll.num[0]*resizefactor # neighbor search grid in x (0)

# Reduce the height of the world and grid
coll.adjustUpperWall()

# Set duration of simulation, automatically determine timestep, set the output
# file interval.
coll.initTemporal(total = 5.0, file_dt = 0.10)

if (collapse == True):

    # Run sphere
    coll.run(dry = True)
    coll.run()

    if (plots == True):
        # Make a graph of the energies
        coll.visualize("energy", savefig=True, outformat='png')

    if (rendering == True):
        # Render images with raytracer with linear velocity as the color code
        print("Rendering images with raytracer")
        coll.render(method = "vel", max_val = 1.0, verbose = False)
        coll.video()
