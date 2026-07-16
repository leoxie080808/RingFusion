# The official files tmf8829.c, tmf8829.h, tmf8829_image.c, and
# tmf8829_image.h are downloaded by tools/install_official_tmf8829_driver.ps1.
idf_component_register(
    SRCS
        "tmf8829.c"
        "tmf8829_image.c"
        "tmf8829_shim.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2c esp_driver_gpio esp_timer
)

# The upstream firmware image uses the Arduino PROGMEM annotation.
# ESP32-C6 memory is directly addressable, so it is intentionally empty here.
target_compile_definitions(${COMPONENT_LIB} PUBLIC PROGMEM=)
