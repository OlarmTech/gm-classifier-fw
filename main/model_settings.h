constexpr int kInputDims = 2;   // input tensor is [kInputDims, kFeatureElementCount]
constexpr int kBatchSize = 1;   // prediction batch size for input
constexpr int kOutputDims = 2;  // output tensor is [kOutputDims, kCategoryCount]

constexpr int kFeatureElementCount = 12;    // number of features in input tensor
constexpr int kCategoryCount = 7;        // number of categories in output tensor
constexpr int kStateCount = 10;         // number of states in state buffer
constexpr int kStateTimeout = 2500;     // global Timeout

constexpr int kOpCount = 4;             // number of operations in NN

// preallocate a certain amount of memory for input, output, and intermediate
// arrays. The size required will depend on the model you are using, and may
// need to be determined by experimentation.
constexpr int kTensorArenaSize = 2 * 1024;
extern const char* kCategoryLabels[kCategoryCount];