import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import argparse
import sys

# Funktion zum Zeichnen von Verbindungen und Gewichten
def draw_connections(ax, positions, show_weights, fontsize):
    for i in range(anzahl_schichten - 1):
        for j in range(len(positions[i])):
            for k in range(len(positions[i + 1])):
                linewidth = np.random.uniform(0.3, 1.5)
                arrow = FancyArrowPatch(positions[i][j], positions[i + 1][k],
                                        connectionstyle="arc3,rad=0.",
                                        arrowstyle="-",
                                        linewidth=linewidth,
                                        alpha=arrow_alpha,
                                        color="black",
                                        mutation_scale=10)
                ax.add_patch(arrow)
                if show_weights:
                    weight = np.round(linewidth / 1.5, 2)
                    random_position = np.random.uniform(0.3, 0.7)
                    mid_point = [positions[i][j][0] + (positions[i + 1][k][0] - positions[i][j][0]) * random_position,
                                 positions[i][j][1] + (positions[i + 1][k][1] - positions[i][j][1]) * random_position]
                    ax.annotate(f"w:{weight}", mid_point, fontsize=fontsize, ha="center", va="center")

# Funktion zum Zeichnen von Neuronen und Biases
def draw_neurons(ax, positions, show_biases, fontsize):
    for i in range(anzahl_schichten):
        for j in range(len(positions[i])):
            circle = plt.Circle(positions[i][j], neuron_radius, color="blue", alpha=0.8)
            ax.add_patch(circle)
            if show_biases:
                bias = np.round(np.random.uniform(0, 1), 2)
                ax.annotate(f"b:{bias}", (positions[i][j][0], positions[i][j][1] - 0.4), fontsize=fontsize, ha="center", va="center")

# Function to calculate total number of parameters in the network
def calculate_parameters(neuronen_pro_schicht):
    total_params = 0
    for i in range(1, len(neuronen_pro_schicht)):
        total_params += (neuronen_pro_schicht[i-1] * neuronen_pro_schicht[i]) + neuronen_pro_schicht[i]
    return total_params

# Hauptfunktion
def main(args):
    global anzahl_schichten
    global arrow_alpha
    global neuron_radius

    # Variablen aus Argumenten
    neuron_radius = args.neuron_radius
    arrow_alpha = args.arrow_alpha
    schicht_abstand = args.schicht_abstand
    fontsize = args.fontsize
    show_weights = args.show_weights
    show_biases = args.show_biases
    neuronen_pro_schicht = args.neuronen_pro_schicht
    output = args.output

    # Berechnungen
    anzahl_schichten = len(neuronen_pro_schicht)
    positions = create_neural_network_positions(neuronen_pro_schicht, schicht_abstand)
    fig, ax = create_plot(positions, schicht_abstand, neuronen_pro_schicht)

    # Draw network
    draw_connections(ax, positions, show_weights, fontsize)
    draw_neurons(ax, positions, show_biases, fontsize)

    # Formatting and display
    ax.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")

    # Optionally export as SVG file
    if output:
        plt.savefig(output, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)

    total_params = calculate_parameters(neuronen_pro_schicht)
    print(f"Total number of parameters in the network: {total_params}")

    plt.show()


# Funktion zum Erstellen neuronaler Netzwerk-Positionen
def create_neural_network_positions(neuronen_pro_schicht, schicht_abstand):
    positions = []
    max_neuronen = max(neuronen_pro_schicht)
    for i in range(len(neuronen_pro_schicht)):
        schicht = []
        n = neuronen_pro_schicht[i]
        yOffset = (max_neuronen - n) / 2
        for j in range(n):
            schicht.append([i * schicht_abstand, j + yOffset])
        positions.append(schicht)
    return positions

# Function to create the plot
def create_plot(positions, schicht_abstand, neuronen_pro_schicht):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, len(positions) * schicht_abstand)
    ax.set_ylim(-1, max(neuronen_pro_schicht))
    ax.patch.set_visible(False)  # Make background transparent
    return fig, ax
    
# Argumente verarbeiten
parser = argparse.ArgumentParser(
    description="Erstelle ein neuronales Netzwerk-Visualisierungsskript.",
    epilog=f"Beispiel: {sys.argv[0]} 4 8 8 8 4 --neuron_radius 0.2 --arrow_alpha 0.5 --schicht_abstand 3 --fontsize 12 --show_weights --show_biases --output=testnet.svg")
parser.add_argument("--neuron_radius", type=float, default=0.2, help="Radius der Neuronen")
parser.add_argument("--arrow_alpha", type=float, default=0.5, help="Transparenz der Verbindungspfeile")
parser.add_argument("--schicht_abstand", type=int, default=3, help="Abstand zwischen den Schichten")
parser.add_argument("--fontsize", type=int, default=8, help="Schriftgröße für Gewichte und Biases")
parser.add_argument("--show_weights", action="store_true", help="Gewichte der Verbindungen anzeigen")
parser.add_argument("--show_biases", action="store_true", help="Biases der Neuronen anzeigen")
parser.add_argument("--output", type=str, help="Name der Ausgabedatei (SVG-Format)")
parser.add_argument("neuronen_pro_schicht", type=int, nargs="+", help="Anzahl der Neuronen pro Schicht")


args = parser.parse_args()

if __name__ == "__main__":
    main(args)


