import matplotlib.pyplot as plt


speedup_data = {
    (10, 10): {2: 0.160, 4: 0.069, 8: 0.026, 16: 0.024},
    (10, 100): {2: 0.479, 4: 0.422, 8: 0.287, 16: 0.150},
    (10, 200): {2: 0.887, 4: 0.409, 8: 0.385, 16: 0.193},
    (10, 500): {2: 1.067, 4: 0.900, 8: 0.544, 16: 0.380},

    (1000, 10): {2: 1.970, 4: 3.407, 8: 5.759, 16: 9.022},
    (1000, 20): {2: 2.000, 4: 3.613, 8: 6.008, 16: 7.458},
    (1000, 50): {2: 2.078, 4: 3.49, 8: 5.891, 16: 8.058},
    (1000, 100): {2: 2.116, 4: 3.377, 8: 5.711, 16: 8.329},
    
    (2000, 10): {2: 2.206, 4: 3.413, 8: 5.849, 16: 7.919},
    (2000, 20): {2: 2.136, 4: 3.546, 8: 6.177, 16: 9.095},
    (2000, 50): {2: 2.228, 4: 3.534, 8: 7.173, 16: 10.030},
    (2000, 100): {2: 2.284, 4: 3.529, 8: 5.869, 16: 10.127},
    
    
    (3000, 5): {2: 2.215, 4: 3.187, 8: 6.349, 16: 8.224},
    (3000, 10): {2: 2.241, 4: 3.621, 8: 6.290, 16: 9.747},
    (3000, 20): {2: 2.246, 4: 3.509, 8: 5.994, 16: 9.981},
    (3000, 50): {2: 2.340, 4: 3.672, 8: 6.962, 16: 10.843},

    (5000, 5): {2: 2.103, 4: 3.602, 8: 6.656, 16: 10.094},
    (5000, 10): {2: 2.184, 4: 3.341, 8: 7.394, 16: 10.770},
    (5000, 20): {2: 2.286, 4: 3.587, 8: 5.997, 16: 10.212733},
    (5000, 50): {2: 2.277, 4: 3.712, 8: 6.025, 16: 10.042}
}


grid_sizes = sorted(set(grid for grid, gens in speedup_data.keys()))

for grid in grid_sizes:
    plt.figure(figsize=(10, 6))
    for (g, gens), proc_data in speedup_data.items():
        if g != grid:
            continue
        processes= sorted(proc_data.keys())
        speedups = [proc_data[p] for p in processes]
        plt.plot(processes, speedups, marker='o', label=f'{gens} generations', linewidth=2)

    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title(f'Speedup vs Processes for Grid {grid}x{grid}')
    plt.xscale('log', base=2) 
    plt.xticks(processes, labels=[str(p) for p in processes])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()