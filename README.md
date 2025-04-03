Simple benchmarking for the distributed sorter from KaDiS (https://github.com/MichaelAxtmann/KaDiS) -- currently only some versions of AMS are benchmarked.

## Compilation
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark
```

## Usage
```bash
  ./build/benchmark --help
```
