CC=nvcc

ARCH=sm_80

LIB_FLAGS=-lm -O3 -arch=$(ARCH) -lcublas 

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN_NAME=main
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).cu

CPU_BIN=test_cpu_only

OBJECTS = $(OBJ_FOLDER)/cpu_gemm.o $(OBJ_FOLDER)/cuda_gemm.o

all: $(BIN_FOLDER)/$(CPU_BIN) $(BIN_FOLDER)/$(MAIN_BIN)

# Compile object file
$(OBJ_FOLDER)/cpu_gemm.o: $(SRC_FOLDER)/cpu_gemm.cpp
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cpu_gemm.cpp -o $@ $(LIB_FLAGS)


#GPU object files

$(OBJ_FOLDER)/cuda_gemm.o: $(SRC_FOLDER)/cuda_gemm.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cuda_gemm.cu -o $@ $(LIB_FLAGS)


# Build final executables
$(BIN_FOLDER)/$(MAIN_BIN): $(SRC_FOLDER)/$(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIB_FLAGS)

$(BIN_FOLDER)/$(CPU_BIN): $(SRC_FOLDER)/test_cpu_only.cpp $(OBJECTS)
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) $^ -o $@ $(LIB_FLAGS) 

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)


                                                                                                                                                                                
