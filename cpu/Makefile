CXX = g++
INCLUDE = -I../util -I../ 
LIBS = -lm -lboost_thread -lboost_system #-lpapi
CFLAGS += -O3 -std=c++11 -fcilkplus -lcilkrts #-DPROFILE -DVALIDATE #-DPAPI_PROFILE 

all: pagerank
build: pagerank

OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp) $(patsubst %.cpp,%.o,$(wildcard ../*.cpp)))

%.o: %.cpp
	$(CXX) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
pagerank: $(OBJ)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf pagerank *.o
