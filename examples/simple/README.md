## Simple

This example simply performs a matrix multiplication, solely for the purpose of demonstrating a basic usage of ggml and backend handling. The code is commented to help understand what each part does.

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
60 & 110 & 54 & 29 \\
55 & 90 & 126 & 28 \\
50 & 54 & 42 & 64 \\
\end{bmatrix}
$$

The `simple-ctx` doesn't support gpu acceleration. `simple-backend` demonstrates how to use other backends like CUDA and Metal.
