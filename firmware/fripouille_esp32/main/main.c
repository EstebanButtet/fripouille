#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "bsp/esp32_s3_touch_lcd_4.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lvgl.h"

#define SERIAL_LINE_MAX 256

LV_FONT_DECLARE(fripouille_font_20);

static const char *TAG = "fripouille";

static lv_obj_t *message_label = NULL;

static void display_set_text(const char *text)
{
    if (!bsp_display_lock(1000)) {
        ESP_LOGE(TAG, "Unable to lock LVGL");
        return;
    }

    lv_label_set_text(message_label, text);
    lv_obj_center(message_label);

    bsp_display_unlock();
}

static void handle_command(const char *line)
{
    if (strcmp(line, "PING") == 0) {
        printf("@PONG\n");
        return;
    }

    if (strncmp(line, "TEXT ", 5) == 0) {
        display_set_text(line + 5);
        printf("@OK TEXT\n");
        return;
    }

    printf("@ERR UNKNOWN_COMMAND\n");
}

static void run_serial_receiver(void)
{
    char line[SERIAL_LINE_MAX];
    size_t line_length = 0;
    bool discard_line = false;
    bool framed_command = false;

    ESP_LOGI(TAG, "USB command receiver ready");

    while (true) {
        char chunk[64];
        ssize_t received = read(STDIN_FILENO, chunk, sizeof(chunk));

        if (received <= 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        for (ssize_t i = 0; i < received; ++i) {
            char c = chunk[i];

            /*
             * '@' is the synchronization marker.
             * Any bytes received before it are ignored.
             */
            if (c == '@') {
                line_length = 0;
                discard_line = false;
                framed_command = true;
                continue;
            }

            if (c == '\r' || c == '\n') {
                if (discard_line) {
                    discard_line = false;
                    framed_command = false;
                    line_length = 0;
                    printf("@ERR LINE_TOO_LONG\n");
                    continue;
                }

                if (!framed_command || line_length == 0) {
                    line_length = 0;
                    framed_command = false;
                    continue;
                }

                line[line_length] = '\0';
                handle_command(line);

                line_length = 0;
                framed_command = false;
                continue;
            }

            if (!framed_command || discard_line) {
                continue;
            }

            if (line_length >= SERIAL_LINE_MAX - 1) {
                discard_line = true;
                continue;
            }

            line[line_length++] = c;
        }
    }
}

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

    message_label = lv_label_create(screen);

    lv_label_set_long_mode(
        message_label,
        LV_LABEL_LONG_MODE_WRAP
    );

    lv_obj_set_width(
        message_label,
        420
    );

    lv_obj_set_style_text_align(
        message_label,
        LV_TEXT_ALIGN_CENTER,
        LV_PART_MAIN
    );

    lv_obj_set_style_text_font(
        message_label,
        &fripouille_font_20,
        LV_PART_MAIN
    );

    lv_label_set_text(
        message_label,
        "FRIPOUILLE"
    );

    lv_obj_set_style_text_color(
        message_label,
        lv_color_hex(0xFFFFFF),
        LV_PART_MAIN
    );

    lv_obj_center(message_label);

    bsp_display_unlock();

    ESP_LOGI(TAG, "Display ready");

    run_serial_receiver();
}
