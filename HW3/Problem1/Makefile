CC = nvcc
TARGET = prob1
OBJECTS = prob1.cu

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) -o $@ $^

clean :
	rm $(TARGET)
