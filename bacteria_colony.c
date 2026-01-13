#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#include <ctype.h>

#define MAX_FILENAME 256

char* grid;
char* new_grid;
char* groundtruth;
int n,m;
int thread_count, nr_gens;

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

void read_grid(char* input_filename, char** local_grid, int* n, int* m, int my_rank, int comm_sz)
{
    char* whole_grid = NULL;

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
        fclose(fin);
    }
        
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = (*n) / comm_sz;
    * local_grid = (char*)malloc(local_rows * (*m) * sizeof(char));

    if (my_rank == 0) {
        whole_grid = (char*)malloc((*n) * (*m) * sizeof(char));
        FILE* fin = fopen(input_filename, "r");
        int dummy_n, dummy_m;
            
        fscanf(fin, "%d %d", &dummy_n, &dummy_m);

        for (int i = 0; i < (*n) * (*m); i++) {
            fscanf(fin, " %c", &whole_grid[i]);
        }
        fclose(fin);
    }

    MPI_Scatter(whole_grid, local_rows * (*m), MPI_CHAR, 
            *local_grid, local_rows * (*m), MPI_CHAR, 
            0, MPI_COMM_WORLD);

    if (my_rank == 0) free(whole_grid);
}

void print_global_grid(char* local_grid, int m, int n, int my_rank, int comm_sz) {
    int local_rows = n / comm_sz;
    char* final_grid = NULL;

    if (my_rank == 0) {
        final_grid = (char*)malloc(n * m * sizeof(char));
    }

    MPI_Gather(local_grid, local_rows * m, MPI_CHAR,
               final_grid, local_rows * m, MPI_CHAR,
               0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%c", final_grid[i * m + j]);
            }
            printf("\n");
        }
        free(final_grid);
    }
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

    if(my_rank == 0)
    {
        get_output_filename(argv[1], output_file);
    }

    
    read_grid(argv[1], &local_slice, &n, &m, my_rank, comm_sz);

    print_global_grid(local_slice,n,m,my_rank,comm_sz);
    free(local_slice);

    MPI_Finalize();
}