
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random
import time
import os
from collections import defaultdict



#  Representación del laberinto

class Maze:
    

    def __init__(self, rows: int, cols: int):
        self.rows = rows          # número de celdas verticales
        self.cols = cols          # número de celdas horizontales
        # Matriz completa con paredes
        self.grid = np.zeros((2 * rows - 1, 2 * cols - 1), dtype=np.int8)
        # Marcar todas las celdas como abiertas
        for r in range(rows):
            for c in range(cols):
                self.grid[2 * r, 2 * c] = 1

    def open_wall(self, r1: int, c1: int, r2: int, c2: int):
        #Abre la pared entre las celdas (r1,c1) y (r2,c2).
        wr = r1 + r2          # fila de la pared en la matriz
        wc = c1 + c2          # columna de la pared en la matriz
        self.grid[wr, wc] = 1

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()

    def neighbors(self, r: int, c: int):
        #Devuelve celdas vecinas válidas (arriba, abajo, izq, der)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append((nr, nc))
        return result



#  Union-Find (para Kruskal)

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True



#  Algoritmo de Kruskal

def kruskal_maze(rows: int, cols: int, seed: int = None):
    
    #Genera un laberinto usando el algoritmo de Kruskal.
    #Retorna el laberinto final y la lista de pasos para animar.
    #Cada paso es una copia del grid en ese momento.
    
    if seed is not None:
        random.seed(seed)

    maze = Maze(rows, cols)

    # Crear lista de todas las paredes (bordes entre celdas adyacentes)
    walls = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                walls.append(((r, c), (r + 1, c)))   # pared horizontal
            if c + 1 < cols:
                walls.append(((r, c), (r, c + 1)))   # pared vertical

    random.shuffle(walls)

    uf = UnionFind(rows * cols)

    def cell_id(r, c):
        return r * cols + c

    steps = []
    steps.append(maze.get_grid())   # estado inicial

    for (r1, c1), (r2, c2) in walls:
        id1, id2 = cell_id(r1, c1), cell_id(r2, c2)
        if uf.union(id1, id2):
            maze.open_wall(r1, c1, r2, c2)
            steps.append(maze.get_grid())

    return maze, steps



#  Algoritmo de Prim

def prim_maze(rows: int, cols: int, seed: int = None):
    
    #Genera un laberinto usando el algoritmo de Prim (versión aleatoria).
    #Retorna el laberinto final y la lista de pasos para animar.

    if seed is not None:
        random.seed(seed)

    maze = Maze(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)

    # Celda inicial aleatoria
    start_r = random.randint(0, rows - 1)
    start_c = random.randint(0, cols - 1)
    visited[start_r, start_c] = True

    # Frontera: lista de (celda_visitada, celda_no_visitada)
    frontier = []
    for nr, nc in maze.neighbors(start_r, start_c):
        frontier.append(((start_r, start_c), (nr, nc)))

    steps = []
    steps.append(maze.get_grid())   # estado inicial

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        (r1, c1), (r2, c2) = frontier[idx]
        frontier.pop(idx)

        if visited[r2, c2]:
            continue

        # Conectar celda nueva con una celda visitada
        maze.open_wall(r1, c1, r2, c2)
        visited[r2, c2] = True
        steps.append(maze.get_grid())

        # Agregar vecinos no visitados a la frontera
        for nr, nc in maze.neighbors(r2, c2):
            if not visited[nr, nc]:
                frontier.append(((r2, c2), (nr, nc)))

    return maze, steps



#  Visualización estática (imagen final)

def plot_maze(maze: Maze, title: str = "Laberinto", ax=None, show: bool = True):
    #Muestra el laberinto final como imagen
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8))

    grid = maze.get_grid()
    display = np.ones_like(grid, dtype=float)   # blanco = pasillo
    display[grid == 0] = 0.0                    # negro = pared

    # Entrada y salida
    rows, cols = maze.rows, maze.cols
    display[0, 0] = 0.5          # entrada (verde más abajo)
    display[2 * rows - 2, 2 * cols - 2] = 0.3   # salida

    cmap = plt.cm.binary
    ax.imshow(display, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

    # Marcar entrada y salida con colores
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 1, 1,
                                     color='#2ecc71', alpha=0.8, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (2 * cols - 2 - 0.5, 2 * rows - 2 - 0.5), 1, 1,
        color='#e74c3c', alpha=0.8, zorder=2))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')

    if standalone and show:
        plt.tight_layout()
        plt.show()

    return ax



#  Comparación lado a lado (imagen estática)

def compare_mazes(rows: int, cols: int, seed: int = 42, output_dir: str = '.'):
    #Genera y muestra ambos laberintos lado a lado
    print(f"Generando laberintos {rows}x{cols}...")

    t0 = time.perf_counter()
    maze_k, steps_k = kruskal_maze(rows, cols, seed=seed)
    t_kruskal = time.perf_counter() - t0

    t0 = time.perf_counter()
    maze_p, steps_p = prim_maze(rows, cols, seed=seed)
    t_prim = time.perf_counter() - t0

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Generación Aleatoria de Laberintos  ({rows}×{cols})",
                 fontsize=16, fontweight='bold', y=1.01)

    plot_maze(maze_k,
              title=f"Kruskal  |  pasos: {len(steps_k)}  |  tiempo: {t_kruskal*1000:.1f} ms",
              ax=axes[0], show=False)
    plot_maze(maze_p,
              title=f"Prim  |  pasos: {len(steps_p)}  |  tiempo: {t_prim*1000:.1f} ms",
              ax=axes[1], show=False)

    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Entrada (1,1)'),
        mpatches.Patch(facecolor='#e74c3c', label=f'Salida ({rows},{cols})'),
        mpatches.Patch(facecolor='black', label='Pared'),
        mpatches.Patch(facecolor='white', label='Pasillo', edgecolor='gray'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'laberintos_comparacion.png'),
                dpi=150, bbox_inches='tight')
    print("Imagen guardada: laberintos_comparacion.png")
    plt.show()

    
    print(f"  Estadísticas de generación ({rows}×{cols})")
    print(f"{'='*50}")
    print(f"  {'Métrica':<25} {'Kruskal':>10} {'Prim':>10}")
    
    print(f"  {'Pasos (paredes abiertas)':<25} {len(steps_k)-1:>10} {len(steps_p)-1:>10}")
    print(f"  {'Tiempo (ms)':<25} {t_kruskal*1000:>10.2f} {t_prim*1000:>10.2f}")
    print(f"  {'Celdas totales':<25} {rows*cols:>10} {rows*cols:>10}")
    

    return maze_k, maze_p, steps_k, steps_p


# ─────────────────────────────────────────────
#  Animación de construcción
# ─────────────────────────────────────────────
def animate_construction(steps: list, title: str, output_file: str,
                          fps: int = 30, max_frames: int = 200):
    
    print(f"Generando animación: {title}  ({len(steps)} pasos totales)...")

    # Submuestrear pasos para no hacer GIFs enormes
    if len(steps) > max_frames:
        indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
        sampled = [steps[i] for i in indices]
    else:
        sampled = steps

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')

    grid0 = sampled[0]
    h, w = grid0.shape

    # Paleta de colores personalizada
    # 0 = pared (oscuro), 1 = pasillo (claro)
    img = ax.imshow(
        np.zeros_like(grid0, dtype=float),
        cmap='Blues', vmin=0, vmax=1,
        interpolation='nearest'
    )

    title_obj = ax.set_title(title, color='white', fontsize=13,
                              fontweight='bold', pad=8)
    step_text = ax.text(0.5, -0.02, '', transform=ax.transAxes,
                        ha='center', color='#a0aec0', fontsize=9)

    total = len(sampled)

    def update(frame):
        grid = sampled[frame]
        display = grid.astype(float)
        img.set_data(display)
        pct = int(100 * frame / (total - 1)) if total > 1 else 100
        step_text.set_text(f"Progreso: {pct}%  |  frame {frame+1}/{total}")
        return img, step_text

    ani = animation.FuncAnimation(
        fig, update, frames=total,
        interval=1000 // fps, blit=True
    )

    ani.save(output_file, writer='pillow', fps=fps,
             savefig_kwargs={'facecolor': '#1a1a2e'})
    plt.close(fig)



#  Comparación animada lado a lado

def animate_comparison(steps_k: list, steps_p: list,
                        rows: int, cols: int,
                        output_file: str = '/mnt/user-data/outputs/construccion_comparacion.gif',
                        fps: int = 20, max_frames: int = 150):

    print(f"Generando animación comparativa...")

    def subsample(steps, n):
        if len(steps) > n:
            idx = np.linspace(0, len(steps) - 1, n, dtype=int)
            return [steps[i] for i in idx]
        return steps

    sk = subsample(steps_k, max_frames)
    sp = subsample(steps_p, max_frames)
    total = max(len(sk), len(sp))

    # Extender la lista más corta repitiendo el último frame
    while len(sk) < total:
        sk.append(sk[-1])
    while len(sp) < total:
        sp.append(sp[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f"Construcción de Laberintos {rows}×{cols}",
                 color='white', fontsize=14, fontweight='bold')

    imgs = []
    subtitles = ['Algoritmo de Kruskal', 'Algoritmo de Prim']
    colors = ['Blues', 'Greens']

    for i, ax in enumerate(axes):
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')
        ax.set_title(subtitles[i], color='#a0d8ef' if i == 0 else '#90ee90',
                     fontsize=12, fontweight='bold')
        im = ax.imshow(np.zeros_like(sk[0], dtype=float),
                       cmap=colors[i], vmin=0, vmax=1,
                       interpolation='nearest')
        imgs.append(im)

    progress_text = fig.text(0.5, 0.01, '', ha='center',
                              color='#a0aec0', fontsize=9)

    def update(frame):
        imgs[0].set_data(sk[frame].astype(float))
        imgs[1].set_data(sp[frame].astype(float))
        pct = int(100 * frame / (total - 1)) if total > 1 else 100
        progress_text.set_text(f"Progreso: {pct}%")
        return imgs + [progress_text]

    ani = animation.FuncAnimation(
        fig, update, frames=total,
        interval=1000 // fps, blit=True
    )

    ani.save(output_file, writer='pillow', fps=fps,
             savefig_kwargs={'facecolor': '#1a1a2e'})
    plt.close(fig)



#  Punto de entrada

if __name__ == '__main__':
    import os

    
    ROWS = 30
    COLS = 40
    SEED = 7

    # ── Carpeta de salida (relativa al script) ─
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print("  Proyecto 2 – IA 2026: Generación de Laberintos")
    print("=" * 55)
    
    # 1. Generar y comparar laberintos (imagen estática)
    maze_k, maze_p, steps_k, steps_p = compare_mazes(ROWS, COLS, seed=SEED,
                                                      output_dir=OUTPUT_DIR)

    # 2. Animación individual de Kruskal
    animate_construction(
        steps_k,
        title=f"Kruskal  –  {ROWS}×{COLS}",
        output_file=os.path.join(OUTPUT_DIR, "kruskal_animacion.gif"),
        fps=25, max_frames=180
    )

    # 3. Animación individual de Prim
    animate_construction(
        steps_p,
        title=f"Prim  –  {ROWS}×{COLS}",
        output_file=os.path.join(OUTPUT_DIR, "prim_animacion.gif"),
        fps=25, max_frames=180
    )

    # 4. Animación comparativa lado a lado
    animate_comparison(
        steps_k, steps_p,
        rows=ROWS, cols=COLS,
        output_file=os.path.join(OUTPUT_DIR, "construccion_comparacion.gif"),
        fps=20, max_frames=150
    )

    print("\nTodos los archivos generados exitosamente.")