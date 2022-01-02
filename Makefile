CXX = g++
LDFLAGS = -lpng
CXXFLAGS = -O3

TARGETS = kmeans_sequential

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
