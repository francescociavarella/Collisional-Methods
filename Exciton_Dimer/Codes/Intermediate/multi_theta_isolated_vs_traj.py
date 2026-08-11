import os
import numpy as np
import matplotlib.pyplot as plt
from plot_style import set_thesis_style, save_fig

set_thesis_style()

# =========================================================
# PARAMETRI DI CONTROLLO
# =========================================================
# Layout: 3 subplot in alto, 2 in basso.
# In basso: 90° e 0° (come richiesto). In alto: 30°, 45°, 60°.
THETA_TOP_DEG = [30.0, 45.0, 60.0]
THETA_BOTTOM_DEG = [0.0, 90.0]

MODE = 'normal'   # 'normal' oppure 'close_to_90'

# Site selector: 0 for |10>, 1 for |01>
site_index = 1
dt = 0.01

# Indice della traiettoria singola da mostrare (colonna di single_trajs)
single_traj_index = 0

# =========================================================
# DIRECTORY CONFIGURATION
# =========================================================
if MODE == 'normal':
    Input_dir = "../../Results/Data/Complete_rho/normal"
elif MODE == 'close_to_90':
    Input_dir = "../../Results/Data/Complete_rho/close_90_deg"
else:
    raise ValueError(f"Unknown mode: {MODE}")

Output_dir = os.path.join("../Results/Plot/Populations", MODE, "Comparison_Theta")
os.makedirs(Output_dir, exist_ok=True)

dt_str = f"{dt:.6f}".replace(".", "p")

label_site = r"$|10\rangle$" if site_index == 0 else r"$|01\rangle$"

# Colore fisso per il sistema isolato (uguale in ogni subplot)
ISOLATED_COLOR = 'black'

# Colore assegnato esplicitamente a ciascun angolo (facilmente modificabile).
# 0° = limite diffusivo -> blu; 90° = limite quantum jump -> rosso/arancione.
# Gli altri angoli hanno colori diversi tra loro, scelti per buona visibilita'.
THETA_COLOR_MAP = {
    0.0:  '#0072B2',   # blu (diffusivo)
    30.0: '#009E73',   # verde
    45.0: '#AE7F26',   # ocra/marrone
    60.0: '#CC79A7',   # rosa/magenta
    90.0: '#D55E00',   # rosso/arancione (quantum jump)
}


# =========================================================
# FUNZIONE DI CARICAMENTO DATI PER UN SINGOLO THETA
# =========================================================
def load_theta_data(theta_target_deg, site_index, single_traj_index):
    theta_rad = np.radians(theta_target_deg)
    theta_str = f"{theta_rad:.6f}".replace(".", "p")

    filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj20000.npz"
    filepath = os.path.join(Input_dir, filename)

    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found. Skipping angle {theta_target_deg}°...")
        return None

    data = np.load(filepath)
    times = data['times']

    pop_iso = data['pop_traj_isolated'][site_index, :]

    raw_pop = data['pop_00'] if site_index == 0 else data['pop_11']
    single_traj = raw_pop[:, single_traj_index]

    data.close()

    return times, pop_iso, single_traj


def plot_single_panel(ax, theta_target_deg, site_index, single_traj_index):
    """
    Disegna, su un singolo asse, la traiettoria collisionale singola e
    quella del sistema isolato per un dato angolo.
    """
    result = load_theta_data(theta_target_deg, site_index, single_traj_index)
    if result is None:
        ax.set_title(f"Dati non trovati ({theta_target_deg}°)")
        return

    times, pop_iso, single_traj = result

    traj_color = THETA_COLOR_MAP.get(theta_target_deg, 'tab:blue')

    ax.plot(times, single_traj, label='Collisional', linewidth=1.5, color=traj_color)
    ax.plot(times, pop_iso, label='Isolated', linewidth=1.2, linestyle='--', color=ISOLATED_COLOR)

    ax.set_xlim(0, 50)

    ax.set_xlabel('Time')
    ax.set_ylabel(r'Population $|1\rangle$')
    ax.legend(title=fr"$\theta = {theta_target_deg}^\circ$", title_fontsize=10, fontsize=9, loc='upper left')


# =========================================================
# FIGURA A GRIGLIA: 3 subplot sopra, 2 subplot sotto, stessa dimensione,
# con i 2 in basso centrati rispetto ai 3 in alto.
# =========================================================
plt.close('all')
fig = plt.figure(figsize=(15, 8))

# Griglia 2x6: ogni subplot (sopra e sotto) occupa 2 colonne su 6,
# quindi tutti i pannelli hanno la stessa larghezza.
# Sopra: colonne [0:2], [2:4], [4:6] -> 3 pannelli affiancati, nessun offset.
# Sotto: colonne [1:3], [3:5] -> 2 pannelli della stessa larghezza (2 colonne),
# spostati di 1 colonna per risultare centrati rispetto ai 3 di sopra.
gs = fig.add_gridspec(2, 6)

ax_top = [fig.add_subplot(gs[0, 0:2]),
          fig.add_subplot(gs[0, 2:4]),
          fig.add_subplot(gs[0, 4:6])]

ax_bottom = [fig.add_subplot(gs[1, 1:3]),
             fig.add_subplot(gs[1, 3:5])]

for ax, theta_deg in zip(ax_top, THETA_TOP_DEG):
    plot_single_panel(ax, theta_deg, site_index, single_traj_index)

for ax, theta_deg in zip(ax_bottom, THETA_BOTTOM_DEG):
    plot_single_panel(ax, theta_deg, site_index, single_traj_index)

save_fig(fig, f"Collisional_vs_Isolated_Grid_dt{dt_str}", Output_dir)

print("Plot a griglia completato e salvato in:", Output_dir)


# =========================================================
# FIGURE SINGOLE: theta = 0° e theta = 90°
# =========================================================
for theta_deg in [0.0, 90.0]:
    plt.close('all')
    fig_single, ax_single = plt.subplots()

    plot_single_panel(ax_single, theta_deg, site_index, single_traj_index)

    theta_deg_str = str(int(theta_deg)) if float(theta_deg).is_integer() else str(theta_deg)
    save_fig(fig_single, f"Collisional_vs_Isolated_Theta{theta_deg_str}_dt{dt_str}", Output_dir)

print("Plot singoli completati e salvati in:", Output_dir)