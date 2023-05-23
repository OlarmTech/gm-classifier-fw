idf_component_register(
    SRCS main.cpp
    REQUIRES "esp_adc" "esp_timer"
    PRIV_REQUIRES tflite-lib
    INCLUDE_DIRS ".")
