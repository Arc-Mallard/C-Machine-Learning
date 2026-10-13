CC = gcc
CFLAGS = -Wall -Wextra -g

TARGET = mdt-eepy-llm
SRCS = main.c $(shell find . -mindepth 2 -name '*.c')
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
