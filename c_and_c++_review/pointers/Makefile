CC = gcc
CFLAGS = -Wall -Wextra -g

# Explicitly list all targets
all: 01.out 02.out 03.out 04.out 05.out 06.out

# Explicit compilation rules
01.out: 01.c
	$(CC) $(CFLAGS) -o 01.out 01.c

02.out: 02.c
	$(CC) $(CFLAGS) -o 02.out 02.c

03.out: 03.c
	$(CC) $(CFLAGS) -o 03.out 03.c

04.out: 04.c
	$(CC) $(CFLAGS) -o 04.out 04.c

05.out: 05.c
	$(CC) $(CFLAGS) -o 05.out 05.c

06.out: 06.c
	$(CC) $(CFLAGS) -o 06.out 06.c

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

run_04: 04.out
	./04.out

run_05: 05.out
	./05.out

run_06: 06.out
	./06.out

# Run all
run: all
	./01.out
	./02.out
	./03.out
	./04.out
	./05.out
	./06.out

.PHONY: all clean run run_01 run_02 run_03 run_04 run_05 run_06
