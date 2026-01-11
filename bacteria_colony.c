#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

//#define DEBUG
char* grid_serial;
char* new_grid;
char* grid_parallel;
char* groundtruth;
int n,m;
int thread_count, nr_gens;

pthread_barrier_t barrier_swap;

void Allocate_and_init(int rank, double **grid, double **new_grid, int local_rows);
void Exchange_frontiers(int rank, int size, double *grid, int local_rows);
void Compute_local(int rank, int size, double *grid, double *new_grid, int local_rows);
void Aggregate_final(int rank, double *grid, int local_rows);

void mem_alloc()
{
    grid_serial = (char*) malloc(n*m*sizeof(char));
    if (!grid_serial)
    {
        printf("Memory allocation error for grid\n");
        exit(1);
    }
    grid_parallel = (char*) malloc(n*m*sizeof(char));
    if (!grid_parallel)
    {
        printf("Memory allocation error for grid\n");
        exit(1);
    }
    new_grid = (char*) malloc(n*m*sizeof(char));
    if (!new_grid)
    {
        printf("Memory allocation error for new grid\n");
        exit(1);
    }
}


int equal_serial()
{
     for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            if (grid_parallel[i * m + j] != grid_serial[i * m + j])
                return 0;
        }
    return 1;
}

void read_grid(char* filename)
{
    FILE *inputfile = fopen(filename,"r");
    if( !inputfile)
    {
        perror("Error reading file");
        exit(1);
    }

    fscanf(inputfile,"%d %d\n", &n, &m);

    mem_alloc();

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<m; j++)
        {
            fscanf(inputfile,"%c",&grid_serial[i*m+j]);
        }
        fscanf(inputfile,"\n");
    }

    memcpy(grid_parallel, grid_serial, n * m * sizeof(char));

}


void write_output(char* filename, char* grid)
{
    FILE *outputfile = fopen(filename,"w");
    if(!outputfile)
    {
        perror("Error writing in file");
        exit(1);
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            fprintf(outputfile, "%c", grid[i*m+j]);
        }
        fprintf(outputfile, "\n");   
    }
}

void extract_file_names(char*inputfile, char* out)
{
    strcpy(out,inputfile);
    char* index=strrchr(out,'.');
    if (index)
    {
        *index='\0';
    }
}

void print_grid(char* grid)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<m; j++)
        { 
            printf("%c ",grid[i*m+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int count_bacteria(int i, int j, char* grid)
{
    
    int count=0;
    for(int k=i-1;k<=i+1;k++)
    {
        if(k<0 || k>=n)
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

void swap_ptr(char **p1, char **p2)
{
    char *tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

void evolve_serial()
{
    #ifdef DEBUG
        printf("Serial grid:\n");
    #endif
    int neighbors;
    for(int k=0; k<nr_gens; k++)
    {
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<m;j++)
            {
                neighbors=count_bacteria(i,j,grid_serial);
                if(grid_serial[i*m+j]=='.' && neighbors==3)
                {
                    new_grid[i*m+j]='X';
                }
                else if(grid_serial[i*m+j]=='X' && (neighbors<2 || neighbors>3))
                {
                    new_grid[i*m+j]='.';
                }
                else
                {
                    new_grid[i*m+j]=grid_serial[i*m+j];
                }
            }
        }
        swap_ptr(&grid_serial, &new_grid);
        #ifdef DEBUG
            print_grid(grid_serial);
        #endif
    }
}

void* evolve_p(void* rank)
{
    int my_rank = *(int *) rank;
    int local_n =  n/thread_count;
    int my_first_row= my_rank*local_n;
    int my_last_row = (my_rank+1) *local_n -1;
    if(my_rank==thread_count-1)
        my_last_row=n-1;

    #ifdef DEBUG
        printf("Parallel grid:\n");
    #endif

    int neighbors;
    for(int k=0; k<nr_gens; k++)
    {
        for(int i=my_first_row; i<=my_last_row; i++)
        {
            for(int j=0; j<m;j++)
            {
                neighbors=count_bacteria(i,j,grid_parallel);
                if(grid_parallel[i*m+j]=='.' && neighbors==3)
                {
                    new_grid[i*m+j]='X';
                }
                else if(grid_parallel[i*m+j]=='X' && (neighbors<2 || neighbors>3))
                {
                    new_grid[i*m+j]='.';
                }
                else
                {
                    new_grid[i*m+j]=grid_parallel[i*m+j];
                }
            }
        }    
        pthread_barrier_wait(&barrier_swap);
        if (my_rank == 0)
            swap_ptr(&grid_parallel, &new_grid);
        pthread_barrier_wait(&barrier_swap);
        #ifdef DEBUG
            print_grid(grid_parallel);
        #endif
    }
    return NULL;
}

void evolve_parallel_1D(int N)
{
    int rank, size;

    if (rank > 0)
    {
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (N % size != 0)
    {
        if (rank == 0)
            printf("N=%d is not divisible by number of processes (size=%d) \n", N, size);
        MPI_Finalize();
        exit(1);
    }

    double *grid, *new_grid;
    int local_rows = N / size;
    int status;

    Allocate_and_init(rank, &grid, &new_grid, local_rows);

    for (int t = 0; t < MAXITER; t++)
    {
        Exchange_frontiers(rank, size, grid, local_rows);

        Compute_local(rank, size, grid, new_grid, local_rows);

        double *tmp = grid;
        grid = new_grid;
        new_grid = tmp;
    }

    Aggregate_final(rank, grid, local_rows);

    free(thread_handles);
    free(tid);
}

void free_mem()
{
    free(new_grid);
    free(grid_serial);
    free(grid_parallel);
}

int main(int argc, char *argv[])
{
    
    if (argc != 4)
    {
        printf("Usage: %s <inputfile> <nr_gens> <nr_threads>\n", argv[0]);
        exit(0);
    }
    thread_count=strtol(argv[3], NULL, 10);
    nr_gens=strtol(argv[2], NULL, 10);
    char serial_file[256], parallel_file[256];

    extract_file_names(argv[1],serial_file);
    extract_file_names(argv[1],parallel_file);
    strcat(serial_file,"_serial_out.txt");
    strcat(parallel_file,"_parallel_out.txt");

    read_grid(argv[1]);
    
    struct timespec start, finish;
    double elapsed_serial,elapsed_parallel;
    printf("Start Serial with %d generations\n", nr_gens);

    clock_gettime(CLOCK_MONOTONIC, &start);
    evolve_serial();
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed_serial = (finish.tv_sec - start.tv_sec);
    elapsed_serial += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed serial time =%lf \n", elapsed_serial);
    write_output(serial_file,grid_serial);
    
    clock_gettime(CLOCK_MONOTONIC, &start);

    evolve_parallel_1D();
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed_parallel = (finish.tv_sec - start.tv_sec);
    elapsed_parallel+= (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed parallel time =%lf \n", elapsed_parallel);

    write_output(parallel_file,grid_parallel);

    if (!equal_serial())
        printf("!!! Parallel version produces a different result! \n");
    else
        printf("Parallel version produced the same result \n");

    printf("Measured Speedup=%f\n ", elapsed_serial / elapsed_parallel);

    free_mem();
}


