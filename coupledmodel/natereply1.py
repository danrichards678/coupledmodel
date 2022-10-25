#%%
from dolfin import *

import leopart as lp

import numpy as np



npart = 15



mesh = UnitSquareMesh(20, 20)

P = FunctionSpace(mesh, "DG", 1)



# Initialize particles

x = lp.RandomCell(mesh).generate(npart)



# Initial particle

a = np.zeros(x.shape[0])

b = np.zeros(x.shape[0])



# Property indices

a_idx, b_idx = 1, 2



p = lp.particles(x, [a, b], mesh)



# Some initial data

a_df_0 = interpolate(Expression("x[0]", degree=1), P)

b_df_0 = interpolate(Expression("x[1]", degree=1), P)

p.interpolate(a_df_0, a_idx)

p.interpolate(b_df_0, b_idx)





def plot_particles(particles):

    positions = particles.positions()

    import matplotlib.pyplot as plt

    idxs_to_plot = (a_idx, b_idx)

    fig, axs = plt.subplots(1, len(idxs_to_plot))

    for prop_idx in idxs_to_plot:

        axs[prop_idx-1].scatter(positions[:,0], positions[:,1],

                                c=particles.get_property(prop_idx), s=0.1)

        axs[prop_idx-1].set_aspect("equal")

        axs[prop_idx-1].set_title(f"particle property {prop_idx}")

    plt.show()





def edit_properties(particles, candidate_cells):

    num_properties = particles.num_properties()

    num_particles_changed = 0


    ind=0
    for c in candidate_cells:

        for pi in range(particles.num_cell_particles(c)):

            particle_props = list(particles.property(c, pi, prop_num)

                                  for prop_num in range(num_properties))



            # Leopart stores particle data in 3D dolfin points

            px = particle_props[0]  # position is always idx 0

            ap = particle_props[a_idx][0]  # a data

            bp = particle_props[b_idx][0]  # b data

            # Property idxs > b_idx are related to the RK advection scheme



            # Reset the particle property data with new values

            particles.set_property(c, pi, a_idx, Point(ind))

            particles.set_property(c, pi, b_idx, Point(5 + ap))
            ind = ind +1


            # Record the number of particles changed

            num_particles_changed += 1



    info(f"Changed {num_particles_changed} particles")





# We change all particles in all cells

all_cells = [c.index() for c in cells(mesh)]



# Direct manipulation method

plot_particles(p)

edit_properties(p, all_cells)

plot_particles(p)
# %%
