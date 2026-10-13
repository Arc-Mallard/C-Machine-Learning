CC = gcc
CFLAGS = -Wall -Wextra -g

TARGET = main
SRCS = $(shell find . -mindepth 2 -name '*.c')
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: clean
