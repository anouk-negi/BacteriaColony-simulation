#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#include <ctype.h>

#define MAX_FILENAME 256
#define DEBUG

char* initial_grid;
char* groundtruth;

int n,m;
int nr_gens;
double start_time; 

void init_grid(char* grid, int n, int m)
{
    for(int i=0; i<n*m; i++)
    {
        grid[i] = initial_grid[i];
    }
}

void get_output_filename(char *input, char *output)
{
    char base[MAX_FILENAME];
    strcpy(base, input);
    //if the input file is f.txt, extract f
    char *dot = strrchr(base,'.');
    if (dot) 
    {
        *dot ='\0';
    }

    sprintf(output, "%s_out.txt", base);
}

void read_init_grid(char* input_filename, int* n, int* m, int my_rank, int comm_sz)
{
    if (my_rank == 0) {
        FILE* fin = fopen(input_filename, "r");
        if(fin == NULL)
        { 
            perror("File error");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (fscanf(fin, "%d %d", n, m) != 2) {
            fprintf(stderr, "Invalid file format\n");
            fclose(fin);
            return;
        }

        initial_grid = (char*) malloc((*n) * (*m) *sizeof(char));
        if(initial_grid == NULL)
        {
            printf("Rank %d: Cannot allocate initial grid\n",my_rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for(int i=0; i< (*n)* (*m); i++)
        {
            fscanf(fin, " %c", &initial_grid[i]);
        }
        fclose(fin);
    }
        
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void print_grid(char* filename, char* grid, int m, int n, int my_rank) {
    int is_stdout = (strcmp(filename, "stdout") == 0);
    
    if (is_stdout || my_rank == 0) {
        FILE* out_stream;
        int should_close = 0;

        if (is_stdout) {
            out_stream = stdout;
        } else {
            out_stream = fopen(filename, "w");
            if (out_stream == NULL) return;
            should_close = 1;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(out_stream, "%c", grid[i * m + j]);
            }
            fprintf(out_stream, "\n");
        }
        fflush(out_stream);
        if (should_close) fclose(out_stream);
    }
}


void allocate_and_init(int rank, char **grid, char **new_grid, int local_rows, int n, int m)
{
    char *whole_grid = NULL;
    if (rank == 0)
    {
        whole_grid = (char*) malloc(n * m * sizeof(char));
        if (whole_grid == NULL)
        {
            printf("Rank %d: Cannot allocate whole grid\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            //to do: deallocate what has already been allocated
        }
        init_grid(whole_grid, n, m);

        #ifdef DEBUG
            printf("Initial grid: \n");
            print_grid("stdout",whole_grid, n, m, rank);
            printf("\n");
            fflush(stdout);
        #endif
    }

    int local_N = local_rows + 2;
    
    *grid = (char*)malloc(m * local_N * sizeof(char));
    *new_grid = (char*)malloc(m * local_N * sizeof(char));
    if ((*grid == NULL) || (*new_grid == NULL))
    {
        printf("Rank %d: Cannot allocate local grid\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }

    MPI_Scatter(whole_grid, local_rows * m, MPI_CHAR, 
            (*grid) + m, local_rows * m, MPI_CHAR, 
            0, MPI_COMM_WORLD);

    #ifdef DEBUG
            printf("Initial chunk of process rank %d \n", rank);
            print_grid("stdout",(*grid)+m, m, local_rows, rank);
            printf("\n");
            fflush(stdout);
    #endif

    if (rank == 0) free(whole_grid);
}

void Exchange_frontiers(int rank, int size, char* grid, int local_rows)
{
    MPI_Status status;

    int top_real = 1 * m;
    int top_halo = 0 * m;
    int bottom_real = local_rows * m;
    int bottom_halo =(local_rows + 1) * m;

    if(rank > 0)
    {
        MPI_Sendrecv(grid + top_real, m, MPI_CHAR, rank - 1, 0,
            grid + top_halo, m, MPI_CHAR, rank - 1, 0,
            MPI_COMM_WORLD, &status);
    }

    if (rank < size - 1)
    {
        MPI_Sendrecv(grid + bottom_real, m, MPI_CHAR, rank + 1, 0,
            grid+ bottom_halo, m, MPI_CHAR, rank + 1, 0,
            MPI_COMM_WORLD, &status);
    }
}

int count_bacteria(int i, int j, int local_n, int m, char* grid)
{
    
    int count=0;
    for(int k=i-1;k<=i+1;k++)
    {
        if(k<0 || k>=local_n)
            continue;
        for(int l=j-1;l<=j+1;l++)
        {
            if(l<0 || l>=m)
                continue;
            if(k==i && l==j)
                continue;
            if(grid[m*k+l]=='X')
            {
                count++;
            }
        }
    }
    return count;
}

void Compute_local(int local_rows,char *grid, char *new_grid)
{
    int neighbours;
    for(int i = 1; i <= local_rows; i++)
    {
        for(int j=0; j< m; j++)
        {
            neighbours = count_bacteria(i, j, local_rows+2, m ,grid);
            if(grid[i*m+j]=='.' && neighbours==3)
            {
                new_grid[i*m+j]='X';
            }
            else if(grid[i*m+j]=='X' && (neighbours<2 || neighbours>3))
            {
                new_grid[i*m+j]='.';
            }
            else
            {
                new_grid[i*m+j]=grid[i*m+j];
            }
        }
    }
}

void aggregate_final(int rank, char* grid, int local_rows, int n, int m, char* output_file)
{
    char* whole_grid = (char*) malloc(sizeof(char) * n * m);
    if(whole_grid == NULL)
    {
        printf("Rank %d: Cannot allocate final big grid\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    MPI_Gather(grid + m, m * local_rows, MPI_CHAR,
         whole_grid, m* local_rows, MPI_CHAR, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        double end_time = MPI_Wtime();
        printf("MPI 1D decomposition time: %f seconds\n", end_time - start_time);
        fflush(stdout);

        print_grid(output_file, whole_grid, m, n, rank);

        #ifdef DEBUG
            printf("Final grid: \n");
            print_grid("stdout", whole_grid, m, n, rank);
        #endif
        
        free(whole_grid);
    }
}


int main(int argc, char *argv[])
{
    int n,m, my_rank, comm_sz, nr_gens;
    char output_file[MAX_FILENAME];
    char* local_slice;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    if(argc < 3) {
        if (my_rank == 0) printf("Usage: mpiexec -n <p> %s <filename> <nr_gens> <\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *grid, *new_grid; 

    read_init_grid(argv[1], &n, &m, my_rank, comm_sz);

    int local_rows = n / comm_sz;
    //allocate local grids
    if(my_rank == comm_sz - 1)
    {
        local_rows +=n % comm_sz;
    }
    allocate_and_init(my_rank,&grid, &new_grid, local_rows, n, m);

    //print_global_grid(initial_grid,n,m,my_rank,comm_sz);
    nr_gens = atoi(argv[2]); //to do: maybe check
    for (int i = 0; i < nr_gens; i++)
    {
        Exchange_frontiers(my_rank, comm_sz, grid, local_rows);

        Compute_local(local_rows,grid, new_grid);
        char *tmp = grid;
        grid = new_grid;
        new_grid = tmp;
    }

    if(my_rank == 0)
    {
        get_output_filename(argv[1], output_file);
    }

    aggregate_final(my_rank, grid, local_rows, n, m, output_file);

    if (my_rank == 0 && initial_grid != NULL) {
        free(initial_grid);
    }

    MPI_Finalize();

}