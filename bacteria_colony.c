#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#include <ctype.h>

#define MAX_FILENAME 256
//#define DEBUG
#define COMPARE_SERIAL

char* initial_grid;
char* groundtruth;

int n,m;
int nr_gens;
double start_time; 

char *compute_serial(char* serial_grid,char* new_grid, int n, int m, int nr_gens);
int equal_grids(char *g1, char *g2);
void init_grid(char* grid, int n, int m);
void get_output_filename(char *input, char *output);
void read_init_grid(char* input_filename, int* n, int* m, int my_rank, int comm_sz);
void print_grid(char* filename, char* grid, int m, int n, int my_rank);
void init_halos(char *grid, int local_rows, int m);
void exchange_frontiers(int rank, int size, char* grid, int local_rows);
int count_bacteria(int i, int j, int local_n, int m, char* grid);


void allocate_and_init(int rank, char **grid, char **new_grid, int local_rows, int n, int m, int comm_sz)
{
    char *whole_grid = NULL;
    int* sendcount = NULL;
    int* displs = NULL;
    if (rank == 0)
    {
        whole_grid = (char*) malloc(n * m * sizeof(char));
        if(whole_grid == NULL)
        {
            printf("Rank %d: Cannot allocate grid.\n",rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    sendcount = (int*)malloc(comm_sz * sizeof(int));
    if(sendcount == NULL)
    {
        free(whole_grid);
        printf("Rank %d: Cannot allocate sendcount.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    displs = (int*)malloc(comm_sz * sizeof(int));
    if(displs == NULL)
    {
        free(whole_grid);
        free(sendcount);
        printf("Rank %d: Cannot allocate displs.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int sum = 0;

    for(int i=0; i < comm_sz; i++)
    {
        int rows = n/comm_sz;
        if(i == comm_sz - 1)
        {
            rows += n%comm_sz;
        }

        sendcount[i] = rows * m;

        displs[i] = sum;
        sum += sendcount[i];
    }

    if (rank == 0) {
        init_grid(whole_grid, n, m);
    }

    #ifdef DEBUG
        printf("Initial grid: \n");
        print_grid("stdout",whole_grid, n, m, rank);
        printf("\n");
        fflush(stdout);
    #endif

    int local_N = local_rows + 2;
    
    *grid = (char*)malloc(m * local_N * sizeof(char));
    if(*grid == NULL)
    {
        if(rank == 0)
        {
            free(whole_grid);
        }
        free(sendcount);
        free(displs);
        printf("Rank %d: Cannot allocate local grids.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    *new_grid = (char*)malloc(m * local_N * sizeof(char));
    if(*new_grid == NULL)
    {
        if(rank == 0)
        {
            free(whole_grid);
        }
        free(sendcount);
        free(displs);
        free(*grid);
        printf("Rank %d: Cannot allocate new local grids.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }

    MPI_Scatterv(whole_grid, sendcount, displs, MPI_CHAR, 
            (*grid) + m, sendcount[rank], MPI_CHAR, 
            0, MPI_COMM_WORLD);

    #ifdef DEBUG
            printf("Initial chunk of process rank %d \n", rank);
            print_grid("stdout",(*grid)+m, m, local_rows, rank);
            printf("\n");
            fflush(stdout);
    #endif

    if (rank == 0)
    {
        free(whole_grid);
    }
    free(sendcount);
    free(displs);
}


void exchange_frontiers(int rank, int size, char* grid, int local_rows)
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

void compute_local(int local_rows,char *grid, char *new_grid)
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

void aggregate_final(int rank, char* grid, int local_rows, int n, int m, char* output_file, int nr_gens, int comm_sz)
{
    char *whole_grid;
    if (rank == 0) {
        whole_grid = (char*) malloc(n * m * sizeof(char));
        if(whole_grid == NULL)
        {
            printf("Rank %d: Cannot allocate grid.\n",rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    int *recvcount = (int*)malloc(comm_sz * sizeof(int));
    if(recvcount == NULL)
    {
        if(rank == 0)
        {
            free(whole_grid);
        }
        printf("Rank %d: Cannot allocate sendcount.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int *displs = (int*)malloc(comm_sz * sizeof(int));
    if(displs == NULL)
    {
        if(rank == 0)
        {
            free(whole_grid);
        }
        free(recvcount);
        printf("Rank %d: Cannot allocate displs.\n",rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int sum =0;

    for(int i=0; i < comm_sz; i++)
    {
        int rows = n/comm_sz;
        if(i == comm_sz - 1)
        {
            rows += n%comm_sz;
        }

        recvcount[i] = rows * m;

        displs[i] = sum;
        sum += recvcount[i];
    }
    
    MPI_Gatherv(grid + m, recvcount[rank], MPI_CHAR,
         whole_grid, recvcount, displs, MPI_CHAR, 0, MPI_COMM_WORLD);


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


        #ifdef COMPARE_SERIAL
            if (initial_grid!=NULL)
            {
            char *serial_grid = (char *)malloc(sizeof(char) * n * m);
            if(serial_grid == NULL)
            {
                free(whole_grid);
                free(recvcount);
                free(displs);
                printf("Rank %d: Cannot allocate serial grid.\n",rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            char *serial_new_grid = (char *)malloc(sizeof(char) * n * m);
            if (serial_new_grid == NULL)
            {
                free(whole_grid);
                free(recvcount);
                free(displs);
                free(serial_grid);
                printf("Rank %d: Cannot allocate serial grid.\n",rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            init_grid(serial_grid, n, m);
            double start_time_s = MPI_Wtime();
            compute_serial(serial_grid, serial_new_grid,n,m,nr_gens);
            double end_time_s = MPI_Wtime();

            printf("Serial elapsed time: %f seconds\n", end_time_s - start_time_s);

            if (!equal_grids(whole_grid, serial_grid))
                printf("Serial and parallel results are different!!!!\n");
            else
                printf("Serial and parallel results are equal.\n");
            free(serial_grid);
            free(serial_new_grid);
        }
        #endif
        free(whole_grid);
    }

    free(recvcount);
    free(displs);
}


int main(int argc, char *argv[])
{
    int my_rank, comm_sz, nr_gens;
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

    read_init_grid(argv[1], &n, &m, my_rank, comm_sz);

    int local_rows = n / comm_sz;
    if(my_rank == comm_sz - 1)
    {
        local_rows += n % comm_sz;
    }

    char *grid, *new_grid; 
    allocate_and_init(my_rank,&grid, &new_grid, local_rows, n, m, comm_sz);

    //making the hallos '.' so they don't interfere with neighbor counting calculation
    init_halos(grid, local_rows, m);
    init_halos(new_grid, local_rows, m);

    //print_global_grid(initial_grid,n,m,my_rank,comm_sz);
    nr_gens = atoi(argv[2]); //to do: maybe check
    for (int i = 0; i < nr_gens; i++)
    {

        exchange_frontiers(my_rank, comm_sz, grid, local_rows);

        compute_local(local_rows,grid, new_grid);
        char *tmp = grid;
        grid = new_grid;
        new_grid = tmp;
    }

    if(my_rank == 0)
    {
        get_output_filename(argv[1], output_file);
    }

    aggregate_final(my_rank, grid, local_rows, n, m, output_file, nr_gens, comm_sz);

    free(grid);
    free(new_grid);
    if (my_rank == 0 && initial_grid != NULL) {
        free(initial_grid);
    }

    MPI_Finalize();

}

void init_grid(char* grid, int n, int m)
{
    for(int i=0; i<n*m; i++)
    {
        grid[i] = initial_grid[i];
    }
}


char* compute_serial(char* serial_grid,char* new_grid, int n, int m, int nr_gens)
{
    int neighbors;
    for(int k=0; k<nr_gens; k++)
    {
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<m;j++)
            {
                neighbors=count_bacteria(i,j,n,m,serial_grid);
                if(serial_grid[i*m+j]=='.' && neighbors==3)
                {
                    new_grid[i*m+j]='X';
                }
                else if(serial_grid[i*m+j]=='X' && (neighbors<2 || neighbors>3))
                {
                    new_grid[i*m+j]='.';
                }
                else
                {
                    new_grid[i*m+j]=serial_grid[i*m+j];
                }
            }
        }
        char *tmp = serial_grid;
        serial_grid = new_grid;
        new_grid = tmp;
    }
    print_grid("stdout",serial_grid, m,n, 0);
}

int equal_grids(char *g1, char *g2)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (g1[i * m + j]!=g2[i * m + j])
                return 0;
    return 1;
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

void init_halos(char *grid, int local_rows, int m)
{
    for(int j=0; j<m; j++)
    {
        grid[j] ='.';
    }

    for(int j=0; j<m; j++)
    {
        grid[(local_rows + 1) * m + j] = '.';
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