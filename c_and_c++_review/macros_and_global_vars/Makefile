CC = gcc
CFLAGS = -Wall -Wextra -g

# Explicitly list all targets
all: 01.out

# Explicit compilation rules
01.out: 01.c
	$(CC) $(CFLAGS) -o 01.out 01.c

# Clean up compiled files
clean:
	rm -f *.out

# Run targets
run_01: 01.out
	./01.out

# Run all
run: all
	./01.out


.PHONY: all clean run run_01

