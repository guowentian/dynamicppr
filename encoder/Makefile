CXX = g++
INCLUDE = -I../util -I../
LIBS = 
CFLAGS += -O3 

all: encoder
build: encoder

OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp) $(patsubst %.cpp,%.o,$(wildcard ../*.cpp)))

%.o: %.cpp
	$(CXX) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
encoder: $(OBJ)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf encoder *.o
