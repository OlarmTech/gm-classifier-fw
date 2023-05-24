#include "NeuralNetwork.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

NeuralNetwork::NeuralNetwork()
{
    error_reporter = new tflite::MicroErrorReporter();

    model = tflite::GetModel(nn_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, 
                             "Model provided is schema version %d not equal to supported version %d.", 
                             model->version(), 
                             TFLITE_SCHEMA_VERSION);
        return;
    }
    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<4>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddSub();
    resolver->AddLogistic();

    tensor_arena = (uint8_t*) malloc(kTensorArenaSize);
    if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(model, 
                                               *resolver, 
                                               tensor_arena, 
                                               kTensorArenaSize);
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    // Make sure the input has the properties we expect.
    if ((input->dims->size != kInputDims) || 
        (input->dims->data[0] != kBatchSize) ||
        (input->dims->data[1] != kFeatureElementCount) || 
        (input->type != kTfLiteFloat32)) 
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
        return;
    }
    output = interpreter->output(0);
    if ((output->dims->size != kOutputDims) || 
        (output->dims->data[0] != kBatchSize) ||
        (output->dims->data[1] != kCategoryCount) || 
        (output->type != kTfLiteFloat32)) 
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model");
        return;
    }
    
}

float* NeuralNetwork::getInputBuffer()
{
    return input->data.f;
}

float* NeuralNetwork::predict()
{
    interpreter->Invoke();
    return output->data.f;
}