CXX = g++
LDFLAGS = -lpng
TARGETS = kmeans_sequential kmeans_omp

kmeans_omp: CXXFLAGS += -fopenmp

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
