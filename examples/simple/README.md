## Simple

This example simply performs a matrix multiplication, solely for the purpose of demonstrating a basic usage of ggml and backend handling. The code is commented to help understand what each part does.

Normally in matrix multiplication...

$$
A * B = C
$$

In `ggml` the weight matrix is already transposed.

So your inputs are: $$A \text{ and } B^T$$  
And the output is: $$C^T$$

Let's look at a traditional matrix multiplication:

$$
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6 \\
\end{bmatrix}
\times
\begin{bmatrix}
10 & 9 & 5 \\
5 & 9 & 4 \\
\end{bmatrix}
\=
\begin{bmatrix}
60 & 90 & 42 \\
55 & 54 & 29 \\
50 &  54 & 28 \\
110 & 126 & 64 \\
\end{bmatrix}
$$

In `ggml` the second matrix above will be stored transposed in memory, and the output will also be transposed in memory.  
So `ggml` will yield:

$$
\begin{bmatrix}
60 & 55 & 50 & 110 \\
90 & 54 & 54 & 126 \\
42 & 29 & 28 & 64 \\
\end{bmatrix}
$$

The `simple-ctx` doesn't support gpu acceleration. `simple-backend` demonstrates how to use other backends like CUDA and Metal.
