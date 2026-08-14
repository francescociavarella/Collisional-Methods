import matplotlib.pyplot as plt
import os

def set_thesis_style():
    """Applica le impostazioni di stile globali per i grafici della tesi."""
    
    # Palette di colori ad alto contrasto (Colorblind-friendly)
    my_palette = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']
    
    plt.rcParams.update({
        # Font e testi
        'font.family': 'serif',      # Serif si sposa meglio con i font di LaTeX (es. Computer Modern)
        'font.size': 11, 
        'axes.titlesize': 13, 
        'axes.labelsize': 11,
        'xtick.labelsize': 11, 
        'ytick.labelsize': 11, 
        'legend.fontsize': 10,
        
        # Dimensioni
        'figure.figsize': (10, 5), 
        'figure.autolayout': True,   # Ottimo, previene testi tagliati
        
        # Griglia disattivata (come da tua richiesta precedente)
        'axes.grid': False,
        
        # Applica i colori personalizzati
        'axes.prop_cycle': plt.cycler(color=my_palette)
    })

import os

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