CC = g++

CCFLAGS = -std=c++2a -O3 -ffast-math -msse4.2 -fopenmp -lm 

all: stencil

stencil: stencil.cpp
	$(CC) $(CCFLAGS) -o $@ $<

run:
	./stencil

clean:
	rm -rf stencil
