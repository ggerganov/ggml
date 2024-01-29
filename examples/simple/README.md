## Simple

This example simply performs a matrix multiplication, solely for the purpose of demonstrating a basic usage of ggml and backend handling. The code is commented to help understand what each part does.

$$
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
\end{bmatrix}
\times
\begin{bmatrix}
10 & 9 \\
5 & 9 \\
\end{bmatrix}
\=
\begin{bmatrix}
92 & 59 \\
82 & 34 \\
\end{bmatrix}
$$

The `simple-ctx` doesn't support gpu acceleration. `simple-backend` demonstrates how to use other backends like CUDA and Metal.
