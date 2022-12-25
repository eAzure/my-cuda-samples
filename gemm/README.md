# GEMM implementation
## implementation

- 0-naive.cu
- 1-wmma.cu
- cublas.cu

## nsight compute

```
# get the bank conflict information
sudo ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./xxx
```