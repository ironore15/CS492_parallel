CC = nvcc
TARGET = prob2
OBJECTS = prob2.cu

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) -o $@ $^

clean :
	rm $(TARGET)
