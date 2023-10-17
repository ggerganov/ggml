---
outline: deep
---

# ggml time api

## ggml init time {#initTime}

ggml init instance prepare for other time api.

- **type**

```c++
  GGML_API void    ggml_time_init(void);
```

- **detail**

Call this once at the beginning of the program

- **example**

```c++

#include "ggml/ggml.h"

void main(){
    ggml_time_init();
    //...
}
```
## ggml_time_ms {#timeMillisecond}

Get time in millisecond.

- **type**

```c++
  GGML_API int64_t ggml_time_ms(void);
```

- **example**

```c++

#include "ggml/ggml.h"

void main(){
    ggml_time_init();
    const auto t_main_start_ms = ggml_time_ms();
    //...
    const auto t_main_end_ms = ggml_time_ms();
    const total = t_main_end_ms-t_main_start_ms;
}
```

## ggml_time_us {#timeMicroseconds}

Get time in microseconds.

- **type**

```c++
  GGML_API int64_t ggml_time_us(void);
```

- **example**

```c++
#include "ggml/ggml.h"

void main(){
    ggml_time_init();
    const auto t_main_start_us = ggml_time_us();
    //...
    const auto t_main_end_us = ggml_time_us();
    const total = t_main_end_us-t_main_start_us;
}
```

