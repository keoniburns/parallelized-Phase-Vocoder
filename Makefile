INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
MPICC = mpicxx
CPP = g++ -std=c++20
CC = gcc
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
MPIFLAGS= -g -Wall -O3 $(INCLUDE_DIRS) 
CFLAGS2= -O3 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
OMPFLAGS = -g -Wall 

PRODUCT= smb mpishift mpiompshift
# MPIFILES= ex2.c
CPPFILES= smb_starter.cpp mpi_pshift.cpp mpiomp_ps.cpp
CFILES=
SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

smb: smb_starter.cpp
	$(CPP) $(CFLAGS2) -o $@ smb_starter.cpp

mpishift: mpi_pshift.cpp
	$(MPICC) $(MPIFLAGS) -o $@ mpi_pshift.cpp -fopenmp

mpiompshift: mpiomp_ps.cpp
	$(MPICC) $(MPIFLAGS) -o $@ mpi_pshift.cpp -fopenmp


