CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -g -std=gnu11

OBJS = mm.o memlib.o

.PHONY: all clean demo

all: $(OBJS)

mm.o: mm.c memlib.h
	$(CC) $(CFLAGS) -c mm.c

memlib.o: memlib.c memlib.h
	$(CC) $(CFLAGS) -c memlib.c

# Compile and run the built-in demo
demo: mm.c memlib.c
	$(CC) $(CFLAGS) -DDEMO -o demo mm.c memlib.c
	./demo

clean:
	rm -f *.o demo
