CC = gcc
CFLAGS = -std=c99 -march=native -msse4a -Ofast -pthread -lm
TARGET = prob1
OBJECTS = prob1.c

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS)

clean :
	rm $(TARGET)
