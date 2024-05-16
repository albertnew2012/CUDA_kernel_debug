
# Kernel Debug Project

## Overview
This project, "kernel_debug", is a CUDA and C++ based application designed for demonstating debug and development of CUDA kernels in vscode with nsight visual studio edition. It utilizes CMake for building the project and targets NVIDIA Turing architecture specifically.

## Prerequisites
- CMake version 3.10 or higher
- CUDA Toolkit (compatible with NVIDIA architecture 75 - Turing)

## Building the Project

To build the project, follow these steps:

1. **Clone the repository:**
   ```
   git clone [repository-url]
   cd [repository-dir]
   ```

2. **Create a build directory:**
   ```
   mkdir build
   cd build
   ```

3. **Run CMake:**
   ```
   cmake ..
   ```

4. **Compile the project:**
   ```
   cmake --build .
   ```

## Running the Application

After building the project, you can run the executable `kernel_debug` generated in the build directory:

```
./kernel_debug
```

## Debugging

The project is configured to compile in debug mode by default, which includes symbols for debugging. This can be useful when using tools like `cuda-gdb` to debug your CUDA kernels.

## Contributing

Feel free to fork the project, make improvements, and submit pull requests. We appreciate your contributions to enhance the debugging capabilities and performance of the application.

## License

[Specify the license under which this project is made available.]
