# MPT-Library
We want to make mpt callable from external tools and languages, thus we wrapp it using SWIG

- Ref: https://github.com/mosaicml/llm-foundry#mpt
- Ref: https://github.com/ggerganov/ggml/tree/master/examples/mpt
- Ref SWIG: https://github.com/swig/swig

## Building
1) We want to build C++ library for various platforms.

```bash
# get this repo
git clone https://github.com/ggerganov/ggml ./ggml
```

- Windows
 ```bash
  docker run -it -v "$pwd/ggml:/ggml" dockcross/windows-static-x64 /bin/bash -c "cd /ggml && rm -rf ./build-win ; mkdir build-win && cd build-win && cmake  -DBUILD_SHARED_LIBS=ON .. && make -j mpt-library"
 ```
 builds into `build-win/bin` files `libggml.dll` and `libmpt-library.dll` with MSVCRT independant compiler - GCC (so will most probably be compatible with any non-dev user windows, runs fast enough given enough ram)
 
 - Linux
 ```bash
 git clone https://github.com/ggerganov/ggml ./ggml-lin
 docker run -it -v "$pwd/ggml:/ggml" dockcross/linux-x64-clang /bin/bash -c "cd /ggml && rm -rf ./build-lin ; mkdir build-lin && cd build-lin && cmake  -DBUILD_SHARED_LIBS=ON -DGGML_FPIC=ON .. && make -j mpt-library"
 ```
 builds `build-lin/examples/mpt-library/libmpt-library.so` 

 - Mac
 
No crosscompilation with docker here
```bash
cd ggml
mkdir build-mac && cd build-mac
cmake -DBUILD_SHARED_LIBS=ON ..
make -j mpt-library
```
builds `build-lin/examples/mpt-library/libmpt-library.dylib`

 2) given librararies will have a simple overloading-inheritance oriented api:
 
 
 
 ```cpp
 struct mpt_params {
    int n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    int seed           = -1; // RNG seed
    int n_predict      = 200; // new tokens to predict, message token size+n_predict must be < n_ctx
    int n_ctx          = 512; // must be > message token size+n_predict

    std::string model      = ""; // model path
	//there are other params for advanced users - see mpt.h
};

 
 struct Mpt {
    Mpt(mpt_params params); // config
    std::string Process(std::string message); // Push in message -> get message+network output
    virtual void OnNewTokenProcessed(std::string token); // overload to get tokens one-by-one while ggml runs
    virtual void OnLogMessage(std::string information); // overload to get logs
	
	//there are other functions for advanced users - see mpt.h...    
};
 ```
 
# Usage:
this example project is intended for use of Mpt as a library in other projects, comes with SWIG wrapped types and functions.

## C++:
1. Connect library and headers to your project and call
```cpp
    // Initialize Mpt parameters
    mpt_params mptParams;
    mptParams.model = "path_to_your_model/model.bin";

    // Create an Mpt instance
    MyMpt mptInstance(mptParams);

    // Process a message
    std::string message = mptInstance.GetRandomMessage(); // Get a random message
    std::cout << "Original message: " << message << std::endl;

    // Tokenize and Process the message
    std::vector<int> tokenizedMessage = mptInstance.TokenizeMessage(message);
    std::string result = mptInstance.ProcessTokenizedMessage(tokenizedMessage);
    
    std::cout << "Processed message: " << result << std::endl;
```

## In other languages:
1. Generate SWIG files for your language (here we will have an example for .Net C# )
```bash
cd ggml\examples\mpt-library
mkdir ./C#/
swig -c++ -csharp -namespace MptLibrary -outdir ./C#/ -o MptWrapper.cxx mpt.i
```
2. build
3. create a project from them (potentially replace `"libmpt_library"` to `"libs/libmpt-library"` in swig generated files given build shared libraries will be stored in `./libs` folder; also some GetEnumerator functions may requier removal yet it is not a problem due to them being about C++ library a code API user will not be directly interacting with)
4. get the model from HuggingFace
5. convert model to FP16 (using mpt example)
6. quantize the model to 5-bits using Q5_0 quantization (using mpt example, as 5 bits models are much smaller and can load into ram faster)
7. run in your favourite language
