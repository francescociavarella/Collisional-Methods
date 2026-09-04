import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import os

# ============================================================
# Mappa colore fisso per angolo theta (in gradi)
# ============================================================
# Convenzione richiesta:
#   theta = 0   -> rosso   (Quantum Jump)
#   theta = 90  -> blu     (State Diffusion)
#   theta = 30  -> rosa
#   theta = 45  -> giallo
#   theta = 60  -> verde
#
# Usata per le curve che rappresentano DATI stocastici legati a un angolo
# specifico (es. Avg_traj, Single_Traj, No-Jump). Le curve teoriche di
# riferimento (Lindblad, Anc_trace) restano invece in nero/grigio, per
# distinguerle visivamente dai dati indipendentemente dall'angolo.
ANGLE_COLORS = {
    0:  '#D51500',  # rosso   - Quantum Jump
    30: '#CC79A7',  # rosa
    45: '#E4B400',  # giallo
    60: '#009E73',  # verde
    90: '#0072B2',  # blu     - State Diffusion
}

# Colore di fallback per angoli non presenti nella mappa (es. angoli custom
# non tra i 5 standard). Grigio scuro neutro, ben distinguibile dal nero
# usato per Lindblad/trace.
DEFAULT_ANGLE_COLOR = '#555555'

# Colore dedicato per l'analisi No-Jump (post-selezione a theta=0). Diverso
# sia dal rosso "Quantum Jump" usato altrove per theta=0, sia dagli altri
# colori della mappa ANGLE_COLORS, cosi' i plot No-Jump restano
# immediatamente distinguibili anche accostati ai plot standard di theta=0.
NO_JUMP_COLOR = '#7B2D8B'  # viola, alto contrasto

# ============================================================
# Sfumature di colore per angolo (per plot con piu' curve legate allo
# stesso angolo, es. varianza per autostato/sito)
# ============================================================
# Per ciascun angolo standard, endpoint (chiaro, scuro) della sfumatura,
# nella stessa famiglia cromatica del colore base in ANGLE_COLORS:
#   theta = 0   -> arancione (chiaro) a rosso scuro
#   theta = 90  -> azzurro (chiaro) a blu scuro
#   theta = 30  -> rosa chiaro a magenta scuro
#   theta = 45  -> giallo chiaro a ocra scuro
#   theta = 60  -> verde chiaro a verde scuro
ANGLE_GRADIENT_ENDPOINTS = {
    0:  ('#FFA500', '#7A0000'),  # arancione -> rosso scuro
    30: ('#F8C8DC', '#8B1A4F'),  # rosa chiaro -> magenta scuro
    45: ('#FFF2A8', '#8A6D00'),  # giallo chiaro -> ocra scuro
    60: ('#A8E6B0', '#00512B'),  # verde chiaro -> verde scuro
    90: ('#87CEEB', '#00264D'),  # azzurro -> blu scuro
}


def _fallback_gradient_endpoints(theta_deg):
    """
    Genera automaticamente due endpoint (chiaro, scuro) per un angolo NON
    tra quelli standard, schiarendo/scurendo il DEFAULT_ANGLE_COLOR nello
    spazio HLS (stessa tonalita', luminosita' diversa).
    """
    base_rgb = mcolors.to_rgb(DEFAULT_ANGLE_COLOR)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    light_rgb = colorsys.hls_to_rgb(h, min(l + 0.35, 0.9), s)
    dark_rgb = colorsys.hls_to_rgb(h, max(l - 0.35, 0.05), s)
    return mcolors.to_hex(light_rgb), mcolors.to_hex(dark_rgb)


def get_angle_gradient(theta_deg, n_colors):
    """
    Restituisce una lista di n_colors colori esadecimali, interpolati
    linearmente tra l'estremo chiaro e quello scuro della sfumatura
    associata a theta_deg (vedi ANGLE_GRADIENT_ENDPOINTS). Utile per
    distinguere piu' curve (es. una per sito/autostato) che appartengono
    tutte allo stesso angolo, mantenendo la coerenza cromatica col resto
    dei plot di quell'angolo.

    Se l'angolo non e' tra quelli standard, gli endpoint sono generati
    automaticamente schiarendo/scurendo DEFAULT_ANGLE_COLOR.

    Parameters:
    - theta_deg : float, angolo in gradi
    - n_colors  : int, numero di colori richiesti (>= 1)

    Returns:
    - colors : list di str, codici colore esadecimali, dal piu' chiaro al
               piu' scuro
    """
    theta_int = int(round(theta_deg))
    if theta_int in ANGLE_GRADIENT_ENDPOINTS:
        light, dark = ANGLE_GRADIENT_ENDPOINTS[theta_int]
    else:
        light, dark = _fallback_gradient_endpoints(theta_deg)

    if n_colors <= 1:
        return [dark]

    cmap = mcolors.LinearSegmentedColormap.from_list('angle_gradient', [light, dark], N=256)
    return [mcolors.to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]


def get_site_colors(n_colors):
    """
    Restituisce una lista di n_colors colori FISSI, uno per sito/autostato,
    INDIPENDENTI dall'angolo theta. Usata quando serve distinguere piu'
    curve appartenenti a siti/autostati diversi nello stesso plot (es. una
    linea per ciascun autostato), mantenendo lo stesso colore per lo stesso
    sito in qualunque angolo — a differenza di get_angle_gradient, che
    invece varia con theta.

    Basata sulla colormap qualitativa 'tab10' di Matplotlib (10 colori ad
    alto contrasto, la stessa gia' usata per distinguere i siti nel plot
    'All populations together'), cosi' i colori restano coerenti tra i vari
    plot del progetto che mostrano piu' siti/autostati contemporaneamente.

    Parameters:
    - n_colors : int, numero di colori richiesti

    Returns:
    - colors : list di str, codici colore esadecimali, uno per sito
               (indice 0 = Sito/Autostato 1, ecc.)
    """
    cmap = plt.get_cmap('tab10')
    return [mcolors.to_hex(cmap(i % 10)) for i in range(n_colors)]


def get_angle_color(theta_deg):
    """
    Restituisce il colore fisso associato all'angolo theta_deg (in gradi),
    secondo la convenzione del progetto (0=rosso/Quantum Jump,
    90=blu/State Diffusion, 30=rosa, 45=giallo, 60=verde).

    Se l'angolo non è tra quelli standard, arrotonda al grado intero più
    vicino e prova comunque il lookup; altrimenti usa DEFAULT_ANGLE_COLOR.

    Parameters:
    - theta_deg : float, angolo in gradi

    Returns:
    - color : str, codice colore esadecimale
    """
    theta_int = int(round(theta_deg))
    return ANGLE_COLORS.get(theta_int, DEFAULT_ANGLE_COLOR)


def set_thesis_style():
    """Applica le impostazioni di stile globali per i grafici della tesi."""
    # Palette di colori ad alto contrasto (Colorblind-friendly), usata per le
    # curve che NON rappresentano dati legati a un angolo specifico (es. per
    # eventuali plot generici che non passano per get_angle_color).
    my_palette = ['#0072B2', "#D51500", "#AE7F26", '#009E73', '#CC79A7']
    plt.rcParams.update({
        # Font e testi — fontsize aumentati per leggibilita'
        'font.family': 'serif',      # Serif si sposa meglio con i font di LaTeX (es. Computer Modern)
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18,
        'lines.linewidth': 2.2,
        # Dimensioni
        'figure.figsize': (10, 5),
        'figure.autolayout': True,   # Ottimo, previene testi tagliati
        # Griglia disattivata (come da richiesta precedente)
        'axes.grid': False,
        # Applica i colori personalizzati di default
        'axes.prop_cycle': plt.cycler(color=my_palette)
    })


def save_fig(fig, filename, output_dir):
    """
    Saves the figure in vector format (PDF).
    Creates the directory automatically if it does not exist.
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Build the full path with the .pdf extension
    path_pdf = os.path.join(output_dir, f"{filename}.pdf")
    # Save the figure without dpi setting, as it is a vector graphic
    fig.savefig(path_pdf, bbox_inches='tight')
    print(f"Saved successfully: {path_pdf}")