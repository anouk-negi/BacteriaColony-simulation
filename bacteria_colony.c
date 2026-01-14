#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#include <ctype.h>

#define MAX_FILENAME 256

char* initial_grid;
char* groundtruth;

int n,m;
int nr_gens;

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


void allocate_and_init(int rank, char **grid, char **new_grid, int n, int m, int comm_sz)
{
    char *whole_grid = NULL;
    if (rank == 0)
    {
        whole_grid = (char*) malloc(n * m * sizeof(char));
        if (whole_grid == NULL)
        {
            printf("Rank %d: Cannot allocate whole grid\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        //init grid
    }

    int local_rows = n / comm_sz;
    //allocate local grids
    if(rank == comm_sz - 1)
    {
        local_rows = n % comm_sz;
    }

    int local_N = local_rows + 2;
    
    *grid = (char*)malloc(n * local_N * sizeof(char));
    *new_grid = (char*)malloc(n * local_N * sizeof(char));
    if ((*grid == NULL) || (*new_grid == NULL))
    {
        printf("Rank %d: Cannot allocate local grid\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //starting parallel

    MPI_Scatter(whole_grid, local_rows * m, MPI_CHAR, 
            (*grid) + m, local_rows * m, MPI_CHAR, 
            0, MPI_COMM_WORLD);

    if (rank == 0) free(whole_grid);
}

void print_global_grid(char* local_grid, int m, int n, int my_rank, int comm_sz) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%c", local_grid[i * m + j]);
            }
            printf("\n");
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int n,m, my_rank, comm_sz;
    char output_file[MAX_FILENAME];
    char* local_slice;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    if(argc < 2) {
        if (my_rank == 0) printf("Usage: mpiexec -n <p> %s <filename>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *grid, *new_grid; 

    /*if(my_rank == 0)
    {
        get_output_filename(argv[1], output_file);
    }*/

    read_init_grid(argv[1], &n, &m, my_rank, comm_sz);

    print_global_grid(initial_grid,n,m,my_rank,comm_sz);

    if (my_rank == 0 && initial_grid != NULL) {
        free(initial_grid);
    }

    MPI_Finalize();

}