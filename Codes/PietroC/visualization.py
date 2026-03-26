import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

import matplotlib.animation as animation
from matplotlib import rcParams


from descriptors.densification import bloch_coords


### FIGURE UTILITIES
def plot_multiple_bloch_trajectories(arrays,
                                     ax = None, fig=None,
                                     colors=['r','b','g'], lw=2, alp=0.75, quiv=True, quiv_init=False, labels=None, title=None, xylabels=False, rot_viev=(15, 45), 
                                     path=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_axis_off()

    # Plot Bloch wireframe
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', linewidth=0.4, alpha=0.5, rstride=10, cstride=10)
    for axis in [[1,0,0], [0,1,0], [0,0,1]]:
        ax.plot([-axis[0], axis[0]], [-axis[1], axis[1]], [-axis[2], axis[2]], color='grey', linewidth=1, alpha=0.5)
    # Labels
    if xylabels:
        ax.text(1.2, 0, 0, "x", fontsize=12)
        ax.text(0, 1.2, 0, "y", fontsize=12)
    ax.text(0, 0, 1.2, r"$|0\rangle$", fontsize=14)
    ax.text(0, 0, -1.3, r"$|1\rangle$", fontsize=14)


    # Plot each trajectory
    print(len(np.array(arrays).shape))
    if len(np.array(arrays).shape) == 3:
        rho_list = arrays
        bloch_vectors = np.array([bloch_coords(rho) for rho in rho_list])
        xs, ys, zs = bloch_vectors.T
        # Path
        ax.plot(xs, ys, zs, color=colors[0], linewidth=lw, alpha=alp, label=labels[0] if labels else None)
        # Arrow at the end
        if quiv:
            arrow_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2], zs[-1] - zs[-2]])
            arrow_len = np.linalg.norm(arrow_vec)
            if arrow_len > 1e-6:  # avoid zero division
                direction = arrow_vec / arrow_len
                scale = 0.2  # fixed arrow length
                ax.quiver(0, 0, 0, xs[-1], ys[-1], zs[-1], color=colors[i], arrow_length_ratio=0.1, linewidth=2)
            if quiv_init:
                ax.quiver(0, 0, 0, xs[0], ys[0], zs[0], color='k', arrow_length_ratio=0.1, linewidth=2)
    
    elif len(np.array(arrays).shape) == 4:
        if len(colors) < np.array(arrays).shape[0]:
            colors = cm.cool(np.linspace(0, 1, np.array(arrays).shape[0])) # plasma
        for i, rho_list in enumerate(arrays):
            bloch_vectors = np.array([bloch_coords(rho) for rho in rho_list])
            xs, ys, zs = bloch_vectors.T
            # Path
            ax.plot(xs, ys, zs, color=colors[i], linewidth=lw, alpha=alp, label=labels[i] if labels and len(labels)<=len(colors) else None)
            # Arrow at the end
            arrow_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2], zs[-1] - zs[-2]])
            arrow_len = np.linalg.norm(arrow_vec)
            
            if quiv:
                #if arrow_len > 1e-6:  # avoid zero division
                    ax.quiver(0, 0, 0, xs[-1], ys[-1], zs[-1], color=colors[i], arrow_length_ratio=0.1, linewidth=2)
                    if quiv_init:
                        ax.quiver(0, 0, 0, xs[0], ys[0], zs[0], color='grey', arrow_length_ratio=0.1, linewidth=2)

    # Axis formatting
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1,1,1])
    # Labels
    if labels:
        ax.legend(loc='upper right')
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    #ax.view_init(elev=30, azim=45)
    ax.view_init(elev=rot_viev[0], azim=rot_viev[1])
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def multi_figure_bloch_plot(n_subplots=2,
                            ncols=2,
                            figsize=(10,10),
                            labels=False,
                            xylabels=False,
                            suptitle=None,):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.95)
        
    # init sphere aesthetics
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    rows = (n_subplots + ncols - 1) // ncols
    
    axes = []
    for i in range(1,n_subplots+1):
        
        ax = fig.add_subplot(rows, ncols, i, projection='3d', )
        
        ax.set_facecolor('white')
        ax.set_axis_off()

        # Plot Bloch wireframe
        ax.plot_wireframe(x, y, z, color='gray', linewidth=0.4, alpha=0.5, rstride=10, cstride=10)
        for axis in [[1,0,0], [0,1,0], [0,0,1]]:
            ax.plot([-axis[0], axis[0]], [-axis[1], axis[1]], [-axis[2], axis[2]], color='grey', linewidth=1, alpha=0.5)
        # Labels
        if xylabels:
            ax.text(1.2, 0, 0, "x", fontsize=12)
            ax.text(0, 1.2, 0, "y", fontsize=12)
        ax.text(0, 0, 1.2, r"$|0\rangle$", fontsize=14)
        ax.text(0, 0, -1.3, r"$|1\rangle$", fontsize=14)

        # Axis formatting
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1,1,1])

        axes.append(ax)
    
    i=ord('a') #chr(97)
    for a in axes:
        a.set_title('({})'.format(chr(i)),  loc='left') # fontsize=fontsize_legend,
        i+=1

    fig.tight_layout()
    return fig, axes

def plot_onebloch_multipletrajectories(ax,
                                       arrays, 
                                       colors=None,#['r','b','g'],
                                       colormap_plain=False,
                                       plaincolor='grey',
                                       lw=2, alp=0.75, 
                                       quiv=True, 
                                       quiv_init=False, 
                                       quiv_color=None,
                                       quiv_alpha=None,
                                       labels=None, title=None, xylabels=False, rot_viev=(15, 45),
                                        showlegend=True):
    """Plot multiple Bloch trajectories in a single 3D plot."""
    
    # Plot each trajectory
    if len(np.array(arrays).shape) == 3:
        if colors:
            if labels and len(colors) < len(labels):
                if colormap_plain:
                    colors = [plaincolor for _ in range(np.array(arrays).shape[0])]
                else:
                    colors = cm.cool(np.linspace(0, 1, np.array(arrays).shape[0])) #cool #plasma #gist_rainbow
        else:
            if colormap_plain:
                colors = [plaincolor for _ in range(np.array(arrays).shape[0])]
            else:
                colors = cm.cool(np.linspace(0, 1, np.array(arrays).shape[0])) #cool #plasma #gist_rainbow
        rho_list = arrays
        bloch_vectors = np.array([bloch_coords(rho) for rho in rho_list])
        xs, ys, zs = bloch_vectors.T
        # Path
        ax.plot(xs, ys, zs, color=colors[0], linewidth=lw, alpha=alp, label=labels[0] if labels else None)
        # Arrow at the end
        if quiv:
            arrow_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2], zs[-1] - zs[-2]])
            arrow_len = np.linalg.norm(arrow_vec)
            if arrow_len > 1e-6:  # avoid zero division
                direction = arrow_vec / arrow_len
                scale = 0.2  # fixed arrow length
                ax.quiver(0, 0, 0,
                            xs[-1], ys[-1], zs[-1],
                            color=quiv_color if quiv_color else colors[i], arrow_length_ratio=0.1, linewidth=2, alpha=quiv_alpha if quiv_alpha else 1.0)
            if quiv_init:
                ax.quiver(0, 0, 0,
                            xs[0], ys[0], zs[0],
                            color='grey', arrow_length_ratio=0.1, linewidth=2)
    
    elif len(np.array(arrays).shape) == 4:
        if colors:
            if len(colors) < np.array(arrays).shape[0]:
                if colormap_plain:
                    colors = [plaincolor for _ in range(np.array(arrays).shape[0])]
                else:
                    colors = cm.cool(np.linspace(0, 1, np.array(arrays).shape[0])) #cool #plasma #gist_rainbow
        else:
            if colormap_plain:
                colors = [plaincolor for _ in range(np.array(arrays).shape[0])]
            else:
                colors = cm.cool(np.linspace(0, 1, np.array(arrays).shape[0])) #cool #plasma #gist_rainbow
        for i, rho_list in enumerate(arrays):
            bloch_vectors = np.array([bloch_coords(rho) for rho in rho_list])
            xs, ys, zs = bloch_vectors.T
            # Path
            ax.plot(xs, ys, zs, color=colors[i], linewidth=2, alpha=alp, label=labels[i] if labels and len(labels)<=len(colors) else None)
            # Arrow at the end
            arrow_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2], zs[-1] - zs[-2]])
            arrow_len = np.linalg.norm(arrow_vec)
            
            if quiv:
                if arrow_len > 1e-6:  # avoid zero division
                    ax.quiver(0, 0, 0,
                                xs[-1], ys[-1], zs[-1],
                                color=quiv_color if quiv_color else colors[i], arrow_length_ratio=0.1, linewidth=2, alpha=quiv_alpha if quiv_alpha else 1.0)
                    if quiv_init:
                        ax.quiver(0, 0, 0,
                                    xs[0], ys[0], zs[0],
                                    color='grey', arrow_length_ratio=0.1, linewidth=2)
    # Labels
    if labels and showlegend:
        ax.legend(loc='upper right')
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=rot_viev[0], azim=rot_viev[1]) 


def add_label_below(fig, axes, box_coord=(0.5,0.), fontsize_legend=12):
    """Add a label below the figure."""
    for i,a in enumerate(axes):
        l = 0
        for j in a.get_legend_handles_labels(): l += len(j)
        if l != 0:
        #if axes[i].get_legend_handles_labels() != ([] for i in range(len(axes))):
            ax_l = a              
    lines_labels = [ax_l.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, 
                ncol=len(labels),
                bbox_to_anchor=box_coord,
                loc='lower center',
                fontsize=fontsize_legend,
            )

    for a in axes:
        a.legend()
        a.get_legend().remove()

    fig.tight_layout()
    plt.show()

def save_figure(fig, path, dpi=300):
    """Save the figure to a file."""
    fig.suptitle('')
    fig.tight_layout()  
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {path}")


 ### ANIMATION UTILITIES

 
def generate_bloch_animation(array, filename="bloch_animation.mp4", color='royalblue', fps=30, showit=True):
    """
    Create a video showing the evolution of a trajectory on the Bloch sphere.

    Parameters:
    - array: (N, 2, 2) array of density matrices
    - filename: output video filename (MP4)
    - color: color of the trajectory
    - fps: frames per second
    """
    # Extract Bloch vector path
    bloch_vectors = np.array([bloch_coords(rho) for rho in array])
    xs, ys, zs = bloch_vectors.T

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)

    # Bloch sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', linewidth=0.4, alpha=0.5, rstride=2, cstride=2)

    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        ax.plot([-axis[0], axis[0]], [-axis[1], axis[1]], [-axis[2], axis[2]],
                color='grey', linewidth=1, alpha=0.5)

    ax.text(1.2, 0, 0, "x", fontsize=10)
    ax.text(0, 1.2, 0, "y", fontsize=10)
    ax.text(0, 0, 1.3, r"$|0\rangle$", fontsize=12)
    ax.text(0, 0, -1.4, r"$|1\rangle$", fontsize=12)

    # Initialize line and arrow
    path_line, = ax.plot([], [], [], color=color, linewidth=2)
    arrow = ax.quiver(0, 0, 0, 0, 0, 0, color=color, arrow_length_ratio=0.1, linewidth=2)

    def update(frame):
        path_line.set_data(xs[:frame], ys[:frame])
        path_line.set_3d_properties(zs[:frame])
        ax.collections.remove(arrow)

        new_arrow = ax.quiver(0, 0, 0,
                              xs[frame], ys[frame], zs[frame],
                              color=color, arrow_length_ratio=0.1, linewidth=2)
        return path_line, new_arrow

    ani = animation.FuncAnimation(fig, update, frames=len(xs), blit=False)
    ani.save(filename, writer='ffmpeg', fps=fps) 
    print(f"Saved animation to {filename}")
    if showit:
        plt.show()
    else:
        plt.close(fig)


def WINDOWS_generate_bloch_animation(array, filename="bloch_animation.gif", color='royalblue', fps=20, showit=True):
    """
    Create a GIF showing the evolution of a trajectory on the Bloch sphere.

    Parameters:
    - array: (N, 2, 2) array of density matrices
    - filename: output file name (should end with .gif)
    - color: color of the trajectory line and arrow
    - fps: frames per second for the animation
    """
    # Convert list of density matrices to Bloch vectors
    coords = np.array([bloch_coords(r) for r in array])
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.axis('off')

    # Draw Bloch sphere
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', linewidth=0.2, alpha=0.3)

    # Axes
    for vec, lbl in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ['x', 'y', r'$|0\rangle$']):
        ax.plot([-vec[0], vec[0]], [-vec[1], vec[1]], [-vec[2], vec[2]], color='gray', linewidth=1)
        ax.text(*np.array(vec) * 1.2, lbl, color='black')

    # Trajectory line and arrow
    path_line, = ax.plot([], [], [], lw=2, color=color)
    arrow = ax.quiver(0, 0, 0, 0, 0, 0, color=color, arrow_length_ratio=0.1, linewidth=2)

    def update(frame):
        nonlocal arrow
        path_line.set_data(xs[:frame], ys[:frame])
        path_line.set_3d_properties(zs[:frame])
        arrow.remove()
        arrow = ax.quiver(0, 0, 0, xs[frame], ys[frame], zs[frame],
                          color=color, arrow_length_ratio=0.1, linewidth=2)
        return path_line, arrow

    ani = animation.FuncAnimation(fig, update, frames=len(xs), blit=False)
    ani.save(filename, writer='pillow', fps=fps)
    print(f"Saved animation to {filename}")
    if showit:
        plt.show()
    else:
        plt.close(fig)



def MULTI_FadingTrails_generate_bloch_animation(rho_list,
                                                filename="bloch.gif",
                                                  fps=20,
                                             colors=None, colormap_use=False, colormap_plain=False, plaincolor='grey',
                                              alp=0.75,
                                              quiv_color=None, quiv_alpha=None,
                                              rot_viev=(15, 45),
                                         trail_len=100, save_every_n=5):
    
    # CONVERT TO LIST OF TRAJECTORIES
    if isinstance(rho_list, (list, np.ndarray)) and isinstance(rho_list[0], np.ndarray) and rho_list[0].shape == (2, 2):
        rho_list = [rho_list]
    num_traj = len(rho_list)
    # COLOR SET
    #colors = colors or [f"C{i}" for i in range(num_traj)]
    if not colors and not colormap_plain and not colormap_use:
        colors = [f"C{i}" for i in range(num_traj)]
    elif colormap_plain:
        colors = [plaincolor] * num_traj
    elif colormap_use:
        colors = cm.cool(np.linspace(0, 1, num_traj))
    
    # EXTRACT BLOCH COORDINATES    
    all_coords = [np.array([bloch_coords(rho) for rho in traj]) for traj in rho_list]
    max_frames = max(len(coords) for coords in all_coords)

    # Downsample frame indices for animation
    frame_indices = list(range(0, max_frames, save_every_n))

    # Initialize the figure and 3D axes
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')

    # Bloch sphere
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    #u = np.linspace(0, 2*np.pi, 100)
    #v = np.linspace(0, np.pi, 100)
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', linewidth=0.4, alpha=0.5, rstride=5, cstride=5)

    # Axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='gray', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], color='gray', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], color='gray', linewidth=1, alpha=0.5)
    ax.text(1.1, 0, 0, 'X', color='gray')
    ax.text(0, 1.1, 0, 'Y', color='gray')
    ax.text(0, 0, 1.1, r'$|0\rangle$', color='gray')

    # choose rotation view
    ax.view_init(elev=rot_viev[0], azim=rot_viev[1]) 

    trails = [[] for _ in range(num_traj)]
    arrows = [ax.quiver(0, 0, 0, 0, 0, 0, color=color, arrow_length_ratio=0.1, linewidth=2) for color in colors]

    def update(frame):
        artists = []
        for i, coords in enumerate(all_coords):
            if frame < len(coords):
                x_now, y_now, z_now = coords[frame]
                arrows[i].remove()
                arrows[i] = ax.quiver(0, 0, 0, x_now, y_now, z_now,
                                      color=quiv_color if quiv_color else colors[i], arrow_length_ratio=0.1, linewidth=2, alpha=quiv_alpha if quiv_alpha else 1.0)
                artists.append(arrows[i])

                for line in trails[i]:
                    line.remove()
                trails[i].clear()

                # Fading trail
                start = max(0, frame - trail_len)
                segment_points = coords[start:frame + 1]
                for j in range(1, len(segment_points)):
                    x_seg = segment_points[j - 1:j + 1, 0]
                    y_seg = segment_points[j - 1:j + 1, 1]
                    z_seg = segment_points[j - 1:j + 1, 2]
                    alpha = alp * (j / len(segment_points))
                    line = Line3D(x_seg, y_seg, z_seg, color=colors[i], alpha=alpha, linewidth=2)
                    ax.add_line(line)
                    trails[i].append(line)
                    artists.append(line)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, blit=False, repeat=False)
    ani.save(filename, writer='pillow', fps=fps)
    #plt.close(fig)
    print(f"Saved animation to {filename}")
    plt.show()