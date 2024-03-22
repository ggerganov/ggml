## Simple

This example simply performs a matrix multiplication, solely for the purpose of demonstrating a basic usage of ggml and backend handling. The code is commented to help understand what each part does.

Traditional matrix multiplication goes like this (multiply row-by-column):

$$
A \times B = C
$$

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

In `ggml`, we pass the matrix $B$ in transposed form and multiply row-by-row. The result $C$ is also transposed:

$$
ggml\\_mul\\_mat(A, B^T) = C^T
$$

$$
ggml\\_mul\\_mat(
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6 \\
\end{bmatrix}
,
\begin{bmatrix}
10 & 5 \\
9 & 9 \\
5 & 4 \\
\end{bmatrix}
)
\=
\begin{bmatrix}
60 & 55 & 50 & 110 \\
90 & 54 & 54 & 126 \\
42 & 29 & 28 & 64 \\
\end{bmatrix}
$$

The `simple-ctx` doesn't support gpu acceleration. `simple-backend` demonstrates how to use other backends like CUDA and Metal.
