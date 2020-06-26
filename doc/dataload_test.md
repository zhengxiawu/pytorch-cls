# Data Loader time test

Here we provide the time loader average time in different epochs, number of workers and backends (custom, dali cpu, dali gpu and torch).

## Server information

``` json
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                80
On-line CPU(s) list:   0-79
Thread(s) per core:    2
Core(s) per socket:    20
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 79
Model name:            Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
Stepping:              1
CPU MHz:               1197.762
CPU max MHz:           3600.0000
CPU min MHz:           1200.0000
BogoMIPS:              4391.72
Virtualization:        VT-x
Hypervisor vendor:     vertical
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              51200K
NUMA node0 CPU(s):     0-19,40-59
NUMA node1 CPU(s):     20-39,60-79
```

## Results

We run Imagenet 2012 val dataloader 3 epochs with 200 batch size and report the average time of each batch:

| CPU(allocate by cluster)| Num Workers        | Backend        | RAM data  | Epoch 1  | Epoch 2  | Epoch 3  | std |
| -----------------       | :---------:        | :---------:    | :-------: |:--------:|:--------:|:--------:|:---:|
| 4                       | 2                  | dali_cpu       | False     |0.5028495 |0.49988992|0.5004478 |Low  |
| 4                       | 2                  | dali_gpu       | False     |0.511625  |0.5153822 |0.50903373|Low  |
| 4                       | 2                  | Torch          | False     |- |-|- |High |
| 4                       | 2                  | custom(opencv) | False     |0.8860033 |0.8708912 |0.8731021 |High |
| 8                       | 2                  | dali_cpu       | False     |0.50517262|0.5111129 |0.5088048 |Low  |
| 8                       | 2                  | dali_gpu       | False     |0.49841730|0.507398  |0.501374  |Low  |
| 8                       | 2                  | Torch          | False     |- |-|- |High |
| 8                       | 2                  | custom(opencv) | False     |0.6996111 |0.685071  |0.6870560 |High |
