CC = gcc
CFLAGS = -std=c99 -march=native -O0 -lm -fopenmp
TARGET = prob2
OBJECTS = prob2.c

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS)

clean :
	rm $(TARGET)
