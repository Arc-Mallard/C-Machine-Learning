CC = gcc
CFLAGS = -Wall -Wextra -g
LDLIBS = -lm

TARGET = mdt-eepy-llm
SRCS = main.c $(shell find . -mindepth 2 -name '*.c')
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDLIBS)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
