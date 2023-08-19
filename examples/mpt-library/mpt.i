/* command 
swig -c++ -csharp -namespace MptLibrary -outdir ./C#/ -o MptWrapper.cxx mpt.i
*/
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"

%module(directors="1") "libmpt_library"

%{
#include "mpt.h"
%}


%feature("director") Mpt;

%template(Layers) std::vector<mpt_layer>;
%template(Buffer) std::vector<char>;
%template(EmbeddingsOfTheTokens) std::vector<gpt_vocab::id>;
%template(Logits) std::vector<float>; //most of the time here
%template(Tokens) std::vector<int>;
%template(LastNTokens) std::vector<int32_t>;
%template(Tensors) std::map<std::string, struct ggml_tensor *>;

%include "mpt.h"
