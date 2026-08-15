#include "bsp/esp32_s3_touch_lcd_4.h"
#include "esp_err.h"
#include "esp_log.h"
#include "lvgl.h"

static const char *TAG = "fripouille";

void app_main(void)
{
    ESP_LOGI(TAG, "Fripouille ESP32 firmware");
    ESP_LOGI(TAG, "Starting display");

    lv_display_t *display = bsp_display_start();

    if (display == NULL) {
        ESP_LOGE(TAG, "Display initialization failed");
        return;
    }

    ESP_ERROR_CHECK(bsp_display_brightness_set(100));

    if (!bsp_display_lock(0)) {
        ESP_LOGE(TAG, "Unable to lock LVGL");
        return;
    }

    lv_obj_t *screen = lv_screen_active();

    lv_obj_set_style_bg_color(
        screen,
        lv_color_hex(0x000000),
        LV_PART_MAIN
    );
    lv_obj_set_style_bg_opa(
        screen,
        LV_OPA_COVER,
        LV_PART_MAIN
    );

    lv_obj_t *label = lv_label_create(screen);
    lv_label_set_text(label, "FRIPOUILLE");

    lv_obj_set_style_text_color(
        label,
        lv_color_hex(0xFFFFFF),
        LV_PART_MAIN
    );

    lv_obj_center(label);

    bsp_display_unlock();

    ESP_LOGI(TAG, "Display ready");
}
