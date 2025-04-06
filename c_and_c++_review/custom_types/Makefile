# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -g
CXXFLAGS = -Wall -Wextra -g -std=c++11

# Explicitly list all targets
all: 01.out 02.out 03.out

# Explicit compilation rules for C files
01.out: 01.c
	$(CC) $(CFLAGS) -o 01.out 01.c

02.out: 02.c
	$(CC) $(CFLAGS) -o 02.out 02.c

# Explicit compilation rule for C++ files (assuming 03 is C++)
03.out: 03.cpp
	$(CXX) $(CXXFLAGS) -o 03.out 03.cpp

# Clean up compiled files
clean:
	rm -f *.out

# Run targets
run_01: 01.out
	./01.out

run_02: 02.out
	./02.out

run_03: 03.out
	./03.out

# Run all
run: all
	./01.out
	./02.out
	./03.out

.PHONY: all clean run run_01 run_02 run_03
