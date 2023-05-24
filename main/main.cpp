#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_settings.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_timer.h"

#define ADC_THRESHOLD 1500  // mV threshold for logic 1 or 0
#define NO_OF_SAMPLES 64    // nuber of sample to average ADC value over

//-----Macros------//
const static char* TAG = "main";
static int adc_raw[2][10];
static int voltage[2][10];

static uint8_t state_counts = 0;
static uint8_t current_state = 0;
static uint8_t previous_state = 0;
static float running_mean = 0;
static float state_buffer[kStateCount]; // stack containing the last 10 states
static bool timeout = false;

//-----Function Declarations------//

static bool adc_calibration_init(adc_unit_t unit, adc_atten_t atten, adc_cali_handle_t* out_handle);

static void adc_calibration_deinit(adc_cali_handle_t handle);

static uint64_t millis();

uint get_current_state(int voltage, int threshold);

void shiftArray(float arr[], int size, int n);

void read_adc1(adc_oneshot_unit_handle_t adc1_handle, adc_cali_handle_t adc1_cali_handle, bool calibrated);

//-----Globals, used for tfLITE------//

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
uint8_t tensor_arena[kTensorArenaSize];
float* model_input_buffer = nullptr;


extern "C" void app_main(void)
{
    //-------------TF Config---------------//
    model = tflite::GetModel(nn_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) 
    {
        TF_LITE_REPORT_ERROR(error_reporter, 
                             "Model provided is schema version %d not equal to supported version %d.", 
                             model->version(), 
                             TFLITE_SCHEMA_VERSION);
        return;
    }
    // pull in all required operations
    static tflite::MicroMutableOpResolver<kOpCount> micro_op_resolver;
    if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) 
    {
        return;
    }
    if (micro_op_resolver.AddMul() != kTfLiteOk) 
    {
        return;
    }
    if (micro_op_resolver.AddSub() != kTfLiteOk) 
    {
        return;
    }
    if (micro_op_resolver.AddLogistic() != kTfLiteOk) 
    {
        return;
    }

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, 
                                                       micro_op_resolver, 
                                                       tensor_arena, 
                                                       kTensorArenaSize);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) 
    {
        TF_LITE_REPORT_ERROR(error_reporter,"AllocateTensors() failed");
        return;
    }
    // Get information about the memory area to use for the model's input.
    model_input = interpreter->input(0);
    if ((model_input->dims->size != kInputDims) || 
        (model_input->dims->data[0] != kBatchSize) || 
        (model_input->dims->data[1] != kFeatureElementCount) ||
        (model_input->type != kTfLiteFloat32)) 
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
        return;
    }
    model_input_buffer = model_input->data.f;
    //-------------End TF Config---------------//

    //-------------ADC1 Init---------------//
    adc_oneshot_unit_handle_t adc1_handle;
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    //-------------ADC1 Config---------------//
    adc_oneshot_chan_cfg_t config = {
        .atten = ADC_ATTEN_DB_11,
        .bitwidth = ADC_BITWIDTH_12,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, ADC_CHANNEL_6, &config));
    //-------------ADC1 Calibration Init---------------//
    adc_cali_handle_t adc1_cali_handle = NULL;
    bool do_calibration = adc_calibration_init(ADC_UNIT_1, ADC_ATTEN_DB_11, &adc1_cali_handle);
    //-------------ADC1 Read---------------//
    ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, ADC_CHANNEL_6, &adc_raw[0][0]));
    ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_handle, adc_raw[0][0], &voltage[0][0]));
    
    // Start sys timer count
    static uint64_t current_time = millis();
    
    while (1) 
    {
        // read multisampled adc value
        read_adc1(adc1_handle, adc1_cali_handle, do_calibration);
        // get current state
        current_state = get_current_state(voltage[0][0], ADC_THRESHOLD);
        
        // timeout
        if (millis() - current_time >= kStateTimeout)
        {
            state_buffer[state_counts] = (float) kStateTimeout / 1000.0; //convert from ms to s
            current_time = millis();
            running_mean += state_buffer[state_counts] * (float) current_state;
            state_counts += 1;
            timeout = true;
            // shift state buffer to be zero filled from the beginning
            shiftArray(state_buffer, kStateCount, kStateCount - state_counts);
        }
        // state change
        else if (current_state != previous_state) 
        {
            state_buffer[state_counts] = (millis() - current_time) / 1000.0;    //pw in seconds
            current_time = millis();
            running_mean += state_buffer[state_counts] * (float) previous_state;
            ESP_LOGI(TAG, "State: %d, Time: %f", previous_state, state_buffer[state_counts]);
            state_counts += 1;
        }


        if (state_counts >= kStateCount || timeout)
        {
            float sum = 0;
            for (int i = 0; i < kStateCount; i++)
            {
                sum += state_buffer[i];
            }
            float mean = running_mean / sum;
 
            //-------------Run Inference---------------//
            // Fill input buffer of tfLite model
            for (int i = 0; i < kStateCount; i++) 
            {
                model_input_buffer[i] = state_buffer[i];
            }
            model_input_buffer[11] = mean;  // set mean
            model_input_buffer[10] = (float) previous_state; //
            //-------------Print Input Buffer---------------//
            ESP_LOGI(TAG, "=======Input Buffer=======");
            for (int i = 0; i < kFeatureElementCount; i++)
            {
                ESP_LOGI(TAG, "Input Data: %f", model_input_buffer[i]);
            }
            // Run the model on the input and make sure it succeeds.
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) 
            {
                TF_LITE_REPORT_ERROR(error_reporter,"Invoke failed");
                return;
            }

            // Obtain a pointer to the output tensor
            TfLiteTensor* output = interpreter->output(0);
            if ((output->dims->size != kOutputDims) ||
                (output->dims->data[0] != kBatchSize) ||
                (output->dims->data[1] != kCategoryCount)) 
                {
                    TF_LITE_REPORT_ERROR(error_reporter, 
                                         "The results for recognition should contain %d elements, but there are %d in an %d-dimensional shape",
                                         kCategoryCount, 
                                         output->dims->data[1],
                                         output->dims->size);
                    return;
                }

            if (output->type != kTfLiteFloat32) 
            {
                TF_LITE_REPORT_ERROR(error_reporter, 
                                     "The results for recognition should be Float32 elements, but are %d",
                                     output->type);
                return;
            }
            float* result_buffer = tflite::GetTensorData<float>(output);

            ESP_LOGI(TAG, "=====Output Prediction=====");
            for (int i = 0; i < kCategoryCount; i++) 
            {
                ESP_LOGI(TAG, "%s: %.3f", kCategoryLabels[i],result_buffer[i]);
            }

            for (size_t i = 0; i < kFeatureElementCount; ++i) { model_input_buffer[i] = 0.0; }
            for (size_t i = 0; i < kStateCount; ++i) { state_buffer[i] = 0.0; }
            running_mean = 0;
            timeout = false;
            state_counts = 0;
        }
        previous_state = current_state;
        // keep the doggo happy
        vTaskDelay(1);
    }
    //Tear Down ADC
    ESP_ERROR_CHECK(adc_oneshot_del_unit(adc1_handle));
    if (do_calibration) 
    {
        adc_calibration_deinit(adc1_cali_handle);
    }

}

static uint64_t millis()
{
    return esp_timer_get_time() / 1000;
}

void read_adc1(adc_oneshot_unit_handle_t adc1_handle, 
               adc_cali_handle_t adc1_cali_handle, 
               bool calibrated)
{
    int adc_avg = 0;
    // read ADC and average over 10 samples
    for (int i = 0; i < NO_OF_SAMPLES; i++)
    {
        ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, ADC_CHANNEL_6, &adc_raw[0][0]));
        adc_avg += adc_raw[0][0];
    }

    adc_avg /= NO_OF_SAMPLES;

    if (calibrated) 
    {
        ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_handle, adc_avg, &voltage[0][0]));
    }
    else
    {
        ESP_LOGE(TAG, "Calibration not done");
    }
}

void shiftArray(float arr[], int size, int n) 
{
    int i, j;
    float temp;

    // Perform n right shifts
    for (i = 0; i < n; i++) 
    {
        // Store the last element
        temp = arr[size - 1];

        // Shift all elements one position to the right
        for (j = size - 1; j > 0; j--) 
        {
            arr[j] = arr[j - 1];
        }

        // Place the last element at the beginning
        arr[0] = temp;
    }
}

uint get_current_state(int voltage, int threshold)
{
    return voltage > threshold ? 1 : 0;
}


/*---------------------------------------------------------------
        ADC Calibration
---------------------------------------------------------------*/
static bool adc_calibration_init(adc_unit_t unit, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;

    if (!calibrated) 
    {
        ESP_LOGI(TAG, "calibration scheme version is %s", "Line Fitting");
        adc_cali_line_fitting_config_t cali_config = {
            .unit_id = unit,
            .atten = atten,
            .bitwidth = ADC_BITWIDTH_DEFAULT,
            .default_vref = 1142,
        };
        ret = adc_cali_create_scheme_line_fitting(&cali_config, &handle);
        if (ret == ESP_OK) {
            calibrated = true;
        }
    }

    *out_handle = handle;
    if (ret == ESP_OK) 
    {
        ESP_LOGI(TAG, "Calibration Success");
    } 
    else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) 
    {
        ESP_LOGW(TAG, "eFuse not burnt, skip software calibration");
    } 
    else 
    {
        ESP_LOGE(TAG, "Invalid arg or no memory");
    }

    return calibrated;
}

static void adc_calibration_deinit(adc_cali_handle_t handle)
{
    ESP_LOGI(TAG, "deregister %s calibration scheme", "Line Fitting");
    ESP_ERROR_CHECK(adc_cali_delete_scheme_line_fitting(handle));
}