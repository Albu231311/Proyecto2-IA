import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from collections import deque
import heapq
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_generator import kruskal_maze, prim_maze


# Colores para visualización
_CMAP = mcolors.ListedColormap([
    '#1a1a2e',  # 0  Pared
    '#f0f0f0',  # 1  Pasillo
    '#4fc3f7',  # 2  Explorado
    '#ff9800',  # 3  Ruta
    '#2ecc71',  # 4  Inicio
    '#e74c3c',  # 5  Meta
])
_NORM = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5, 6], ncolors=6)

WALL, PASSAGE, EXPLORED, PATH_VAL, START_VAL, GOAL_VAL = 0, 1, 2, 3, 4, 5


# Contenedor de resultados de búsqueda
class SearchResult:
    def __init__(self, path, explored, time_ms):
        self.path = path
        self.explored = explored
        self.time_ms = time_ms
        self.nodes_explored = len(explored)
        self.path_length = len(path)


# Navegación por el laberinto: vecinos, reconstrucción de camino, etc.

def _neighbors(maze, r, c):
    """Yield cells reachable from (r,c) — wall between them must be open."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.rows and 0 <= nc < maze.cols:
            if maze.grid[r + nr, c + nc] == 1:
                yield (nr, nc)


def _reconstruct(came_from, goal):
    """Walk came_from back from goal to start (came_from[start] == None)."""
    if goal not in came_from:
        return []
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# Algoritmos de busqueda

def bfs(maze, start, goal):
    t0 = time.perf_counter()
    queue = deque([start])
    came_from = {start: None}
    explored = []
    while queue:
        cur = queue.popleft()
        explored.append(cur)
        if cur == goal:
            break
        for nb in _neighbors(maze, *cur):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)
    return SearchResult(_reconstruct(came_from, goal), explored,
                        (time.perf_counter() - t0) * 1000)


def dfs(maze, start, goal):
    t0 = time.perf_counter()
    stack = [(start, None)]
    came_from = {}
    explored = []
    visited = set()
    while stack:
        cur, parent = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        came_from[cur] = parent
        explored.append(cur)
        if cur == goal:
            break
        for nb in _neighbors(maze, *cur):
            if nb not in visited:
                stack.append((nb, cur))
    return SearchResult(_reconstruct(came_from, goal), explored,
                        (time.perf_counter() - t0) * 1000)


def ucs(maze, start, goal):
    t0 = time.perf_counter()
    counter = 0
    heap = [(0, counter, start)]
    came_from = {start: None}
    g = {start: 0}
    closed = set()
    explored = []
    while heap:
        cost, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        explored.append(cur)
        if cur == goal:
            break
        for nb in _neighbors(maze, *cur):
            new_cost = cost + 1
            if nb not in g or new_cost < g[nb]:
                g[nb] = new_cost
                came_from[nb] = cur
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nb))
    return SearchResult(_reconstruct(came_from, goal), explored,
                        (time.perf_counter() - t0) * 1000)


def astar(maze, start, goal):
    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    t0 = time.perf_counter()
    counter = 0
    heap = [(h(start), 0, counter, start)]
    came_from = {start: None}
    g = {start: 0}
    closed = set()
    explored = []
    while heap:
        _, cost, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        explored.append(cur)
        if cur == goal:
            break
        for nb in _neighbors(maze, *cur):
            ng = cost + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came_from[nb] = cur
                counter += 1
                heapq.heappush(heap, (ng + h(nb), ng, counter, nb))
    return SearchResult(_reconstruct(came_from, goal), explored,
                        (time.perf_counter() - t0) * 1000)


# Helpers para visualización

def _base_display(maze):
    g = maze.get_grid()
    return np.where(g == 1, PASSAGE, WALL).astype(float)


def _paint_explored(display, cells, maze):
    for r, c in cells:
        display[2 * r, 2 * c] = EXPLORED
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < maze.rows and 0 <= nc < maze.cols:
                if maze.grid[r + nr, c + nc] == 1:
                    display[r + nr, c + nc] = EXPLORED


def _paint_path(display, path):
    for r, c in path:
        display[2 * r, 2 * c] = PATH_VAL
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        display[r1 + r2, c1 + c2] = PATH_VAL


def _paint_endpoints(display, start, goal):
    display[2 * start[0], 2 * start[1]] = START_VAL
    display[2 * goal[0], 2 * goal[1]] = GOAL_VAL


# Visualización fija

def plot_solution(maze, result, title, ax=None, show=True):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))

    display = _base_display(maze)
    _paint_explored(display, result.explored, maze)
    _paint_path(display, result.path)
    if result.path:
        _paint_endpoints(display, result.path[0], result.path[-1])

    ax.imshow(display, cmap=_CMAP, norm=_NORM, interpolation='nearest')
    ax.set_title(
        f"{title}\n"
        f"Camino: {result.path_length}  |  "
        f"Explorados: {result.nodes_explored}  |  "
        f"Tiempo: {result.time_ms:.2f} ms",
        fontsize=10, fontweight='bold', pad=6,
    )
    ax.axis('off')

    if standalone and show:
        plt.tight_layout()
        plt.show()
    return ax


# Visualización animada

def animate_solver(maze, result, title, output_file, fps=20, max_frames=200):
    print(f"  Animando: {title}  ({len(result.explored)} nodos explorados)...")

    if not result.path:
        print("  No se encontró camino, omitiendo animación.")
        return

    explored = result.explored
    path = result.path
    start, goal = path[0], path[-1]

    # Exploración
    explore_cap = max(10, max_frames - 40)
    if len(explored) > explore_cap:
        milestones = np.linspace(0, len(explored) - 1, explore_cap, dtype=int).tolist()
    else:
        milestones = list(range(len(explored)))
    explore_frames = len(milestones)

    # Ruta
    path_steps = min(30, len(path))
    path_milestones = np.linspace(0, len(path) - 1, path_steps, dtype=int).tolist()
    path_frames = len(path_milestones)

    hold_frames = 15
    total_frames = explore_frames + path_frames + hold_frames

    display = _base_display(maze)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)

    img = ax.imshow(display.copy(), cmap=_CMAP, norm=_NORM, interpolation='nearest')
    info = ax.text(0.5, -0.03, '', transform=ax.transAxes,
                   ha='center', color='#a0aec0', fontsize=10)

    last_exp = [0]
    last_path_idx = [0]

    def update(frame):
        if frame < explore_frames:
            up_to = milestones[frame] + 1
            for i in range(last_exp[0], up_to):
                r, c = explored[i]
                display[2 * r, 2 * c] = EXPLORED
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < maze.rows and 0 <= nc < maze.cols:
                        if maze.grid[r + nr, c + nc] == 1:
                            display[r + nr, c + nc] = EXPLORED
            last_exp[0] = up_to
            pct = int(100 * frame / explore_frames)
            info.set_text(f"Explorando... {pct}%  ({up_to} nodos)")

        elif frame < explore_frames + path_frames:
            pf = frame - explore_frames
            up_to = path_milestones[pf] + 1
            for i in range(last_path_idx[0], up_to):
                r, c = path[i]
                display[2 * r, 2 * c] = PATH_VAL
                if i > 0:
                    pr, pc = path[i - 1]
                    display[pr + r, pc + c] = PATH_VAL
            last_path_idx[0] = up_to
            _paint_endpoints(display, start, goal)
            info.set_text(f"Trazando camino... ({up_to}/{len(path)} celdas)")

        else:
            _paint_endpoints(display, start, goal)
            info.set_text(
                f"Camino: {result.path_length} celdas  |  "
                f"Explorados: {result.nodes_explored}  |  "
                f"Tiempo: {result.time_ms:.2f} ms"
            )

        img.set_data(display)
        return img, info

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000 // fps, blit=True)
    ani.save(output_file, writer='pillow', fps=fps,
             savefig_kwargs={'facecolor': '#1a1a2e'})
    plt.close(fig)
    print(f"  Guardado: {os.path.basename(output_file)}")


# Comparación de algoritmos

_ALGORITHMS = [
    ('BFS', bfs),
    ('DFS', dfs),
    ('UCS', ucs),
    ('A*',  astar),
]


def compare_solvers(maze, start, goal, output_dir='.'):
    print(f"\n  Resolviendo laberinto {maze.rows}x{maze.cols}  ({start} -> {goal})")
    print(f"{'=' * 62}")
    print(f"  {'Algoritmo':<10} {'Camino':>8} {'Explorados':>12} {'Tiempo (ms)':>13}")
    print(f"  {'-'*45}")

    results = {}
    for name, algo in _ALGORITHMS:
        res = algo(maze, start, goal)
        results[name] = res
        print(f"  {name:<10} {res.path_length:>8} {res.nodes_explored:>12} {res.time_ms:>13.2f}")

    # 2×2 comparativa visual
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f"Comparación de Algoritmos de Búsqueda  —  {maze.rows}×{maze.cols}",
        color='white', fontsize=15, fontweight='bold', y=1.01,
    )

    for ax, (name, _) in zip(axes.flat, _ALGORITHMS):
        ax.set_facecolor('#1a1a2e')
        res = results[name]
        display = _base_display(maze)
        _paint_explored(display, res.explored, maze)
        _paint_path(display, res.path)
        if res.path:
            _paint_endpoints(display, res.path[0], res.path[-1])
        ax.imshow(display, cmap=_CMAP, norm=_NORM, interpolation='nearest')
        ax.set_title(
            f"{name}\n"
            f"Camino: {res.path_length}  |  "
            f"Explorados: {res.nodes_explored}  |  "
            f"{res.time_ms:.2f} ms",
            color='white', fontsize=11, fontweight='bold',
        )
        ax.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Entrada'),
        mpatches.Patch(facecolor='#e74c3c', label='Salida'),
        mpatches.Patch(facecolor='#4fc3f7', label='Explorado'),
        mpatches.Patch(facecolor='#ff9800', label='Camino'),
        mpatches.Patch(facecolor='#1a1a2e', label='Pared', edgecolor='gray'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=11, bbox_to_anchor=(0.5, -0.02),
               facecolor='#2d2d4e', labelcolor='white')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'solver_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\n  Imagen guardada: solver_comparison.png")
    plt.show()

    return results


# Main
def main():
    ROWS, COLS = 60, 80
    SEED = 42
    START = (0, 0)
    GOAL = (ROWS - 1, COLS - 1)

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("  Proyecto 2 – IA 2026: Solución de Laberintos (Inciso 2)")
    print("=" * 60)

    print("\nGenerando laberinto 60×80 con Kruskal (seed=42)...")
    maze, _ = kruskal_maze(ROWS, COLS, seed=SEED)

    # 1. Imagen comparativa estática (corre los 4 algoritmos)
    results = compare_solvers(maze, START, GOAL, output_dir=OUTPUT_DIR)

    # 2. GIFs individuales para cada algoritmo
    gif_configs = [
        ('BFS', 'bfs_solucion.gif'),
        ('DFS', 'dfs_solucion.gif'),
        ('UCS', 'ucs_solucion.gif'),
        ('A*',  'astar_solucion.gif'),
    ]

    print()
    for name, fname in gif_configs:
        animate_solver(
            maze, results[name],
            title=f"{name}  —  {ROWS}×{COLS}",
            output_file=os.path.join(OUTPUT_DIR, fname),
            fps=20, max_frames=200,
        )

    print("\nTodos los archivos generados exitosamente.")

main()
