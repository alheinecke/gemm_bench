CC=icc
CFLAGS=-O3 -I$(LIBXSMM_DIR)/include -xHost -qopenmp
LDFLAGS=-L$(LIBXSMM_DIR)/lib -lxsmm -lxsmmext -lpthread

LIBXSMM_DIR=/nfs_home/dmudiger/my_libxsmm_gh

all: bench ibench

bench: bench.c $(LIBXSMM_DIR)/include/libxsmm.h
	$(CC) $(CFLAGS) bench.c $(LDFLAGS) -o bench

ibench: ibench.c $(LIBXSMM_DIR)/include/libxsmm.h
	$(CC) $(CFLAGS) ibench.c $(LDFLAGS) -o ibench

clean: 
	rm -rf bench
