CXX = g++
INCLUDE = -I../ -I../util
LIBS = 
CFLAGS += -O3 

all: workload
build: workload

OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp) $(patsubst %.cpp,%.o,$(wildcard ../*.cpp)))

%.o: %.cpp
	$(CXX) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
workload: $(OBJ)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf workload *.o
