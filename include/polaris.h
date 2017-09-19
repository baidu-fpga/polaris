/**
 * @file      polaris.h
 *
 * @brief     declaration device APIs
 *
 * @date      2017-08-07
 *
 * @authors   isa@baidu.com
 *
 * @copyright (C) 2017 Baidu, Inc
 */

#ifndef BAIDU_POLARIS_INCLUDE_POLARIS_H
#define BAIDU_POLARIS_INCLUDE_POLARIS_H

#include <stdint.h>
#include <string.h>
#define FPGACTL_NODE_DEVNAME "fpgactl"

/**
 * @defgroup datatype Data Types
 * All data types and enums of Polaris library.
 */

/**
 * @defgroup context Context Management
 * Functions related to operations of PolarisContext.
 */

/**
 * @defgroup memory Memory Management
 * Functions related to memory allocation, free and copy, including memcpy between CPU and FPGA
 * and kinds of on-FPGA memcpy.
 */

/**
 * @defgroup compute Computation Interface
 * Functions that involve computation. This group provides the core computation ability of
 * Polaris library.
 */

/**
 * @defgroup query Device Management
 * Functions related to device management, including quering the version and matching devices.
 */

/**
 * @addtogroup datatype
 * @{ */

/**
 * Indicates the copy direction
 */
typedef enum {
    /// Copy from FPGA to CPU
    POLARIS_DEVICE_TO_HOST = 0,

    /// Copy from CPU to FPGA
    POLARIS_HOST_TO_DEVICE = 1,

    /// Copy from FPGA to CPU
    POLARIS_DEVICE_TO_DEVICE = 2,
} PolarisMemcpyKind;

/**
 * Indicates whether transpose operation needs to be performed
 */
typedef enum {
    /// Non-transpose operation
    POLARIS_NO_TRANS = 0,

    /// Transpose operation
    POLARIS_TRANS = 1
} PolarisTransType;

/**
 * Types to specify the data precision
 */
typedef enum {
    /// 32-bit floating-point
    POLARIS_FP32 = 0,

    /// 16-bit floating-point
    POLARIS_FP16,

    /// 32-bit signed integer
    POLARIS_INT32,

    /// 16-bit signed integer
    POLARIS_INT16,

    /// 8-bit signed integer
    POLARIS_INT8
} PolarisDataType;

/**
 * Data format supported by convolution operation
 */
typedef enum {
    /// NHWC format
    POLARIS_FORMAT_NHWC = 0,

    /// NCHW format
    POLARIS_FORMAT_NCHW = 1,
} PolarisDataFormat;

/**
 * Function types supported by pooling operation
 */
typedef enum {
    /// Max-pooling
    POLARIS_POOLING_MAX = 0,

    /// Avg-pooling
    POLARIS_POOLING_AVG = 1,
} PolarisPoolingMode;

/// @cond IGNORED

/**
 * Function types supported by elementwise operation
 */
typedef enum {
    /// Illegal type
    POLARIS_ELEMENTWISE_ILLEGAL = 0,

    /// Element-wise softsign activation
    POLARIS_ELEMENTWISE_SOFTSIGN,

    /// Element-wise softsign deactivation
    POLARIS_ELEMENTWISE_DSOFTSIGN,

    /// Vector-vector operation: y := a*x + y
    POLARIS_ELEMENTWISE_AXPY,

    /// Element-wise tanh activation
    POLARIS_ELEMENTWISE_TANH,

    /// Element-wise tanh deactivation
    POLARIS_ELEMENTWISE_DTANH,

    /// Element-wise vsum
    POLARIS_ELEMENTWISE_VSUM,

    /// Element-wise memset
    POLARIS_ELEMENTWISE_MEMSET,

    /// Element-wise sigmoid activation
    POLARIS_ELEMENTWISE_SIGMOID,

    /// Element-wise relu activation
    POLARIS_ELEMENTWISE_RELU,

    /// Element-wise multiply
    POLARIS_ELEMENTWISE_MUL,

    /// Element-wise minimal
    POLARIS_ELEMENTWISE_MIN,

    /// Element-wise maximal
    POLARIS_ELEMENTWISE_MAX,
} PolarisElementwiseFunctionType;

/// @endcond

/**
 * Activation types
 */
typedef enum {
    /// Non-activation
    POLARIS_NO_ACTIVATION = 0,

    /// Tanh activation
    POLARIS_TANH = POLARIS_ELEMENTWISE_TANH,

    /// Sigmoid activation
    POLARIS_SIGMOID = POLARIS_ELEMENTWISE_SIGMOID,

    /// Relu activation
    POLARIS_RELU = POLARIS_ELEMENTWISE_RELU,
} PolarisActivationType;

/**
 * Conv-stream types
 */
typedef enum {
    /// Convmtx operation
    POLARIS_CONVSTREAM_CONVMTX1D = 0,

    /// Deconvmtx operation
    POLARIS_CONVSTREAM_DCONVMTX1D,

    /// maxpooling operation
    POLARIS_CONVSTREAM_MAXPOLLING1D,

    /// dmaxpooling operation
    POLARIS_CONVSTREAM_DMAXPOLLING1D,
} PolarisConvStreamFunctionType;

/** @} */

/************
 * CONSTANTS
 ************/
enum {
    /// max device number
    MAX_DEVICE_COUNT = 64,

    /// max length of version string
    MAX_VERSION_STRING_LENGTH = 32,

    /// max length of firmware-name string
    MAX_FIRMWARE_NAME_LENGTH = 32,
};


/**
 * @ingroup datatype
 * @brief Context object
 *
 * The data structure that holds polaris library context
 */
typedef struct {
    /// Device id
    int devid;

    /// file descriptor to the device
    int fd;
} PolarisContext;

/**
 * @ingroup datatype
 * Return status of polaris interface
 */
typedef enum {
    /// Everything goes fine
    POLARIS_OK = 0,

    /// The requested operation is not supported
    POLARIS_ERR_NOT_SUPPORT = 1,

    /// Invalid paramaters
    POLARIS_ERR_INVALID = 2,

    /// Runtime error
    POLARIS_ERR_RUNTIME = 3,
} PolarisStatus;

/**
 * @ingroup datatype
 * Function types supported by polaris_eltwise()
 */
typedef enum {
    /// element-wise addition
    POLARIS_ADD = POLARIS_ELEMENTWISE_AXPY,

    /// element-wise multiplication
    POLARIS_MUL = POLARIS_ELEMENTWISE_MUL,

    /// element-wise minimal of two vectors
    POLARIS_MIN = POLARIS_ELEMENTWISE_MIN,

    /// element-wise maximal of two vecotrs
    POLARIS_MAX = POLARIS_ELEMENTWISE_MAX,
} PolarisElementWiseType;

/**
 * @ingroup context
 * @brief Create a polaris context object on a specific device.
 *
 * The device id can be obtained by calling polaris_get_devices().
 * Once the context is created, it cannot be moved to another device.
 *
 * @param[in] devid   Device id
 *
 * @return Pointer to a PolarisContext object, NULL for failure
 *
 * @see     polaris_get_devices()
 */
PolarisContext* polaris_create_context(int devid);

/**
 * @ingroup context
 * @brief Free and destroy a polaris context
 *
 * @param[in] ctxt   Pointer to a PolarisContext object
 */
void polaris_destroy_context(PolarisContext* ctxt);

/**
 * @ingroup memory
 * @brief Allocate a block of memory on FPGA.
 *
 * The allocated memory block is associated with the Context object, it will be auto
 * released if the context is destroyed. But there is no boundary in the access of memory
 * allocated in different contexts, that is to say, one context could read/write the
 * content of the memory allocated in another context.
 *
 * @param[in]   ctxt   pointer to PolarisContext object
 * @param[in]   size   size of memory in bytes to allocate
 * @param[out]  ptr    malloced FPGA memory, NULL on error
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_malloc(PolarisContext* ctxt, size_t size, void** ptr);

/**
 * @ingroup memory
 * @brief Allocate a block of memory on host(CPU) memory.
 *
 * CPU memory allocated by polaris_malloc_host() is faster than that of glibc malloc()
 * when performing polaris_memcpy(), it is highly recommended to use this function to
 * allocate memory for those blocks that need to be frequently copied to or from
 * FPGA memory.
 *
 * @note Memory allocated by this function will be pinned in the physical memory,
 * so avoid using this function for CPU-use-only memory.
 *
 * @param[in]  ctxt   pointer to PolarisContext object
 * @param[in]  size   number of bytes to allocate
 * @param[out] ptr    malloced CPU memory address
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_malloc_host(PolarisContext* ctxt, size_t size, void** ptr);

/**
 * @ingroup memory
 * @brief Free a block of memory on FPGA.
 *
 * You can only free those memories that were allocated in the same context.
 *
 * @param[in] ctxt  pointer to PolarisContext object
 * @param[in] ptr   start address of the FPGA memory to be freed
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_free(PolarisContext* ctxt, void* ptr);

/**
 * @ingroup memory
 * @brief Free a block of memory on host(CPU) memory.
 *
 * @param[in] ctxt   pointer to PolarisContext object
 * @param[in] ptr    start address of the CPU memory to be freed
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_free_host(PolarisContext* ctxt, void* ptr);

/**
 * @ingroup memory
 * @brief Copy block of memory between CPU and FPGA.
 *
 * Copy @p size bytes from the memory area pointed to by @p src to the memory area pointed to
 * by @p dest, where @p kind specifies the direction of the copy, and must by one of
 * #POLARIS_DEVICE_TO_HOST, #POLARIS_HOST_TO_DEVICE, #POLARIS_DEVICE_TO_DEVICE
 *
 * @param[in]  ctxt    pointer to PolarisContext object
 * @param[in]  kind    type of transfer
 * @param[out] dest    destination FPGA/CPU memory address
 * @param[in]  src     source CPU/FPGA memory address
 * @param[in]  size    number of bytes to be copied
 *
 * @return Execution status, #POLARIS_OK on success.
 *
 * @see PolarisMemcpyKind
 */
PolarisStatus polaris_memcpy(PolarisContext* ctxt, PolarisMemcpyKind kind,
                             void* dest, const void* src, size_t size);
/**
 * @ingroup compute
 * @brief Perform matrix multiplication with bias and activation support.
 *
 * This function perform the calculation as the formula:
 *
 *     c = activation( alpha * op(a) * op(b) + beta * c + bias )
 *
 * This function behaves similar with `SGEMM` function in
 * [CBLAS](http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html), with the extended
 * ability of adding bias vector and performing activation functions, which match better for the
 * need of deep neural network. In another word, this function provides the ability of the forward
 * process of a full-connected layer.
 *
 * Parameters like @p trans_a, @p trans_b, @p m, @p n, @p k, @p alpha, @p a, @p lda @p b, @p ldb,
 * @p beta, @p c, @p ldc all have the same meaning with that in CBLAS `SGEMM`.
 *
 * @p bias is a vector with dimension `1 * n`, which will be added to each row of output
 * matrix @p c.
 *
 * @p activation indicates the activation function type to be performed after the bias procession.
 *
 * Caller could specify the data type of input matrix @p a, @p b, @p c with @p type_a, @p type_b,
 * @p type_c , data type of @p bias is the same with @p c.
 *
 * @note this function best performed with @p trans_a equals to #POLARIS_NO_TRANS and @p trans_b
 * equals to #POLARIS_TRANS.
 *
 * @warning @p lda, @p ldb and @p ldc is currently not supported in this version of Polaris,
 *          this functionality will be released later in December.\n
 *          For data type, only POLARIS_FP32 is available for now, support for other data type
 *          will be released later.
 *
 * @param[in]     ctxt         pointer to PolarisContext object
 * @param[in]     trans_a      if #POLARIS_NO_TRANS, @p a is `m * k`, otherwise `k * m`
 * @param[in]     trans_b      if #POLARIS_NO_TRANS, @p b is `n * k`, otherwise `k * n`
 * @param[in]     m            dimension m
 * @param[in]     n            dimension n
 * @param[in]     k            dimension k
 * @param[in]     alpha        scalar parameter
 * @param[in]     a            FPGA address of matrix a
 * @param[in]     type_a       data type of matrix a
 * @param[in]     lda          same as LDA in BLAS
 * @param[in]     b            FPGA address of matrix b
 * @param[in]     type_b       data type of matrix b
 * @param[in]     ldb          same as LDB in BLAS
 * @param[in]     beta         scalar parameter
 * @param[in,out] c            FPGA address of matrix c
 * @param[in]     type_c       data type of matrix c and bias
 * @param[in]     ldc          same as LDC in BLAS
 * @param[in]     bias         FPGA address of bias
 * @param[in]     activation   activaion type
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_gemm(PolarisContext* ctxt,
                           PolarisTransType trans_a, PolarisTransType trans_b,
                           int m, int n, int k,
                           float alpha,
                           const void* a, PolarisDataType type_a, int lda,
                           const void* b, PolarisDataType type_b, int ldb,
                           float beta,
                           void* c, PolarisDataType type_c, int ldc,
                           const void* bias,
                           PolarisActivationType activation);

/**
 * @ingroup memory
 * @brief Fill a range of FPGA memory with zero.
 *
 * This function fills the first @p size bytes of the FPGA memory area pointed to by @p ptr with 0.
 *
 * @param[in] ctxt   pointer to PolarisContext object
 * @param[in] ptr    address of the FPGA memory to be set
 * @param[in] size   size of the memory
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_memset(PolarisContext* ctxt, void* ptr, size_t size);

/**
 * @ingroup compute
 * @brief Calculate activation functions.
 *
 * This function calculate the following formula:
 *
 *     dest = alpha * ActivationType(src) + beta * dest
 *
 * where both @p src and @p dest have @p length elements. Activation type is indicated by @p type.
 *
 * @param[in]   ctxt     pointer to PolarisContext object
 * @param[in]   type     activation type
 * @param[in]   length   number of elements in input
 * @param[in]   alpha    alpha value
 * @param[in]   src      input matrix
 * @param[in]   beta     beta value
 * @param[out]  dest     output matrix
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_activation(PolarisContext* ctxt, PolarisActivationType type, size_t length,
                                 float alpha, const float* src, float beta, float* dest);

/**
 * @ingroup compute
 * @brief Calculate derived activation functions.
 *
 * This function calculate the following formula:
 *
 *     dest_diff = DerivedActivation(src, src_diff, dest)
 *
 * it is used in the backward phase, and calculate @p dest_diff according to @p src, @p src_diff and
 * @p dest, where @p dest and @p src is actually the input and output data seperately of the forward
 * phase of the same operation, while @p src_diff is the gradient of @p src.
 *
 * @param[in]  ctxt        pointer to PolarisContext
 * @param[in]  type        activation type
 * @param[in]  length      number of elements in input
 * @param[in]  alpha       alpha value
 * @param[in]  src         backward src data (output data of the forward phase)
 * @param[in]  src_diff    gradient of @p src
 * @param[in]  dest        backward destination data (input data of the forward phase)
 * @param[in]  beta        beta value
 * @param[out] dest_diff   gradient of @p dest
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_deactivation(PolarisContext* ctxt, PolarisActivationType type, size_t length,
                                   float alpha, const float* src, const float* src_diff,
                                   const float* dest, float beta, float* dest_diff);

/**
 * @ingroup compute
 * @brief Element wise functions.
 *
 * Perform element-wise operations indicated by the following fomula:
 *
 *     c = ElementWiseOperation(alpha0 * a, alpha1 * b) + beta * c
 *
 * where @p type indicates the exact operation to be performed.
 *
 * - #POLARIS_ADD `:= v0 + v1`
 * - #POLARIS_MUL `:= v0 * v1`
 * - #POLARIS_MIN `:= min(v0, v1)`
 * - #POLARIS_MAX `:= max(v0, v1)`
 *
 * where `v0` is `alpha0 * a` and `v1` is `alpha1 * b`.
 *
 * @param[in]  ctxt     pointer to PolarisContext
 * @param[in]  type     element wise operation type
 * @param[in]  length   number of elements in input
 * @param[in]  alpha0   scalar parameter
 * @param[in]  a        input matrix
 * @param[in]  alpha1   scalar parameter
 * @param[in]  b        input matrix
 * @param[in]  beta     scalar parameter
 * @param[out] c        output matrix
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_eltwise(PolarisContext* ctxt, PolarisElementWiseType type, size_t length,
                              float alpha0, const float* a,
                              float alpha1, const float* b,
                              float beta, float* c);

/**
 * @ingroup compute
 * @brief Perform matrix transpose on FPGA.
 *
 * Dimension of matrix @p src is `m * n`, and @p dest is `n * m`.
 *
 * @warning @p dest and @p src should **NOT** be the same.
 *
 * @param[in]  ctxt   pointer to PolarisContext object
 * @param[in]  m      dimension m
 * @param[in]  n      dimension n
 * @param[in]  src    FPGA address of input matrix
 * @param[out] dest   FPGA address of output matrix
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_transpose(PolarisContext* ctxt, int m, int n, const void* src, void* dest);

/**
 * @ingroup memory
 * @brief Perform multiple FPGA memcpy in a specific pattern.
 *
 * Copy @p copy_count block of memorys, the length of each memory block is indicated by
 * @p copy_length, the source and destination address of i-th copy
 * (i = 0, 1, ..., @p copy_count -1) is `src[i * src_step + src_padding]` and
 * `dest[i * dest_step + dest_padding]` seperately.
 *
 * @param[in]  ctxt          pointer to PolarisContext
 * @param[in]  copy_count    number of memory blocks to be copied
 * @param[in]  copy_length   length of each memory copy
 * @param[out] dest          destination FPGA address
 * @param[in]  dest_step     destination step
 * @param[in]  dest_padding  destination padding
 * @param[in]  src           source FPGA address
 * @param[in]  src_step      source step
 * @param[in]  src_padding   source padding
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_batch_memcpy(PolarisContext* ctxt, int copy_count, size_t copy_length,
        void* dest, size_t dest_step, size_t dest_padding,
        const void* src, size_t src_step, size_t src_padding);

/* todo: next version
PolarisStatus polaris_matrix_split(PolarisContext* ctxt, size_t output_count, int m, int input_n,
        const int* output_ns, const void* input, void** outputs);

PolarisStatus polaris_matrix_merge(PolarisContext* ctxt, size_t input_count, int m, const int* input_ns,
        int output_n, const void** inputs, void* output);
*/

/**
 * @ingroup query
 * @brief Get list of all the FPGA devices
 *
 * This function always returns the number of all FPGA devices installed in the system, even if
 * it is greater than @p devs_len, but in anycase, it will never write more than @p devs_len
 * device ids into @p devs.
 *
 * @param[out] devs      list of all FPGA devices' ids
 * @param[in]  devs_len  malloced length (number of ints) of @p devs
 *
 * @return Non-negtive value for total FPGA device count,
 *         `-errno` if error happens.
 **/
int polaris_get_devices(int* devs, int devs_len);

/**
 * @ingroup query
 * @brief Get list of all the FPGA devices that match the given firmware name
 *
 * This function always returns the number of all FPGA devices that match the @p firmware_name,
 * even if it is greater than @p devs_len, but in anycase, it will never write more than @p devs_len
 * device ids into @p devs.
 *
 * @param[in]  firmware_name   firmware name to query
 * @param[out] devs            list of all FPGA devices' ids
 * @param[in]  devs_len        malloced length (number of ints) of @p devs
 *
 * @return Non-negtive value for matched FPGA device count,
 *         `-errno` if error happens.
 **/
int polaris_get_devices(const char* firmware_name, int* devs, int devs_len);

/**
 * @ingroup query
 * @brief Get version string of the driver
 *
 * This function always returns the length of driver version string (including the trailing '\0'),
 * even if it is greater than @p ver_len, but it will never write more than @p ver_len chars
 * into @p ver.
 *
 * @param[out]  ver      version string buffer
 * @param[in]   ver_len  malloced length (number of bytes) for @p ver
 *
 * @return Non-negtive value for actual version string length (including the trailing '\0')
 *         `-errno` if error happens.
 **/
int polaris_get_driver_version(char* ver, int ver_len);

/**
 * @brief Read a FPGA register.
 *
 * @warning Make sure you know what you are doing when using this interface, most of the developers
 *          do not have to use this interface.
 *
 * @param[in]  ctxt    pointer to context
 * @param[in]  addr    register address
 * @param[out] value   where to store the read
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_read_register(PolarisContext* ctxt, uint64_t addr, uint64_t *value);

/**
 * @brief Write a FPGA register.
 *
 * @warning Make sure you know what you are doing when using this interface, most of the developers
 *          do not have to use this interface.
 *
 * @param[in]  ctxt   pointer to context
 * @param[in]  addr   register address
 * @param[in] value   the value to write
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_write_register(PolarisContext* ctxt, uint64_t addr, uint64_t value);

#endif
