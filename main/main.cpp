#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_timer.h"

//-----Macros------//
const static char *TAG = "main";
static int adc_raw[2][10];
static int voltage[2][10];

static uint8_t state_counts = 0;
static uint8_t current_state = 0;
static uint8_t previous_state = 0;
static float running_mean = 0;
static float state_buffer[kStateCount]; // stack containing the last 10 states
static bool timeout = false;

//-----Function Declarations------//

static bool adc_calibration_init(adc_unit_t unit, adc_atten_t atten, adc_cali_handle_t *out_handle);

static void adc_calibration_deinit(adc_cali_handle_t handle);

uint get_current_state(int voltage, int threshold);

void shiftArray(float arr[], int size, int n);

void print_output(float* result);

//-----Globals, used for tfLITE------//

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
tflite::ErrorReporter *error_reporter = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
float* model_input_buffer = nullptr;


extern "C" void app_main(void)
{
    //-------------TF Config---------------//

    // pull in all required operations

    // Build an interpreter to run the model with.

    // Allocate memory from the tensor_arena for the model's tensors.

    // Get information about the memory area to use for the model's input.

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
    
    while (1) {
        int adc_avg = 0;
        // read ADC and average over 10 samples
        for (int i = 0; i<10; i++){

        }
        adc_avg /= 10;

        if (do_calibration) {

        }
        
        // timeout
        if (esp_timer_get_time() / 1000 - current_time >= kStateTimeout){

        }
        // state change
        else if (current_state != previous_state) {

        }
        else {
        }

        if (state_counts >= kStateCount || timeout){
 
            //-------------Run Inference---------------//
            // Fill input buffer of tfLite model

            //-------------Print Input Buffer---------------//

            // Run the model on the input and make sure it succeeds.


            // Obtain a pointer to the output tensor and read the predicted label

            //-------------Print Output Buffer---------------//
 
            // reset values
        }
        previous_state = current_state;
        // keep the doggo happy
        vTaskDelay(1);
    }
    //Tear Down ADC
    ESP_ERROR_CHECK(adc_oneshot_del_unit(adc1_handle));
    if (do_calibration) {
        adc_calibration_deinit(adc1_cali_handle);
    }

}


/*---------------------------------------------------------------
        ADC Calibration
---------------------------------------------------------------*/
static bool adc_calibration_init(adc_unit_t unit, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;

    if (!calibrated) {
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
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Calibration Success");
    } else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) {
        ESP_LOGW(TAG, "eFuse not burnt, skip software calibration");
    } else {
        ESP_LOGE(TAG, "Invalid arg or no memory");
    }

    return calibrated;
}

static void adc_calibration_deinit(adc_cali_handle_t handle)
{
    ESP_LOGI(TAG, "deregister %s calibration scheme", "Line Fitting");
    ESP_ERROR_CHECK(adc_cali_delete_scheme_line_fitting(handle));
}