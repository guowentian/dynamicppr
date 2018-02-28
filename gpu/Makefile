NVCC = nvcc
INCLUDE = -I./ -I.. -I../util -I../../cub -I../cpu
LIBS = #-lm 
CFLAGS += -O3 -lcusparse -gencode arch=compute_52,code=sm_52 -gencode arch=compute_35,code=sm_35 #-DPROFILE -DVALIDATE -DNVPROFILE

all: pagerank
build: pagerank

OBJ = $(patsubst %.cu,%.o,$(wildcard *.cu)) $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(patsubst %.cpp,%.o,$(wildcard ../*.cpp))

%.o: %.cpp 
	$(NVCC) $< $(CFLAGS) $(INCLUDE) -c -o $@

%.o: %.cu
	$(NVCC) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
pagerank: $(OBJ)
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf pagerank *.o
