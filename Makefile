# Makefile

CC=g++
NVCC=nvcc

CFLAGS=
NVCCFLAGS	= -ccbin g++ -Xcompiler "-std=c++11"
LIBS		= -lcuda -lcudart

INCLUDES=-I*.h

CPP_SRCS = main.cpp lodepng.cpp helpers.cpp
CU_SRCS  = kernels.cu 

OBJS = $(CPP_SRCS:.cpp=.o)
OBJS += $(CU_SRCS:.cu=.o)

TARGET = filter

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LIBS) -o $(TARGET) 

.SUFFIXES: 

.SUFFIXES:  .cpp .cu .o

.cu.o:
	$(NVCC) -o $@ -c $<

.cpp.o:
	$(NVCC) -o $@ -c $<

clean:
	rm -rf $(TARGET) *.o
