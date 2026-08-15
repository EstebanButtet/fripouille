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
static lv_obj_t *left_eye = NULL;
static lv_obj_t *right_eye = NULL;
static lv_obj_t *mouth = NULL;

#define FACE_NEON_COLOR 0x53F3FF

static void style_neon_outline(
    lv_obj_t *object,
    int32_t width,
    int32_t height
)
{
    lv_obj_set_size(
        object,
        width,
        height
    );

    lv_obj_set_style_bg_opa(
        object,
        LV_OPA_TRANSP,
        LV_PART_MAIN
    );

    lv_obj_set_style_border_width(
        object,
        4,
        LV_PART_MAIN
    );

    lv_obj_set_style_border_color(
        object,
        lv_color_hex(FACE_NEON_COLOR),
        LV_PART_MAIN
    );

    lv_obj_set_style_radius(
        object,
        LV_RADIUS_CIRCLE,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_color(
        object,
        lv_color_hex(FACE_NEON_COLOR),
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_width(
        object,
        22,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_spread(
        object,
        2,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_opa(
        object,
        180,
        LV_PART_MAIN
    );

    lv_obj_clear_flag(
        object,
        LV_OBJ_FLAG_SCROLLABLE
    );
}


static void create_neon_pupil(
    lv_obj_t *eye
)
{
    lv_obj_t *pupil = lv_obj_create(
        eye
    );

    lv_obj_set_size(
        pupil,
        15,
        29
    );

    lv_obj_set_style_bg_color(
        pupil,
        lv_color_hex(0xD9FCFF),
        LV_PART_MAIN
    );

    lv_obj_set_style_bg_opa(
        pupil,
        LV_OPA_COVER,
        LV_PART_MAIN
    );

    lv_obj_set_style_border_width(
        pupil,
        0,
        LV_PART_MAIN
    );

    lv_obj_set_style_radius(
        pupil,
        LV_RADIUS_CIRCLE,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_color(
        pupil,
        lv_color_hex(FACE_NEON_COLOR),
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_width(
        pupil,
        12,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_opa(
        pupil,
        200,
        LV_PART_MAIN
    );

    lv_obj_clear_flag(
        pupil,
        LV_OBJ_FLAG_SCROLLABLE
    );

    lv_obj_center(
        pupil
    );
}


static void create_face(
    lv_obj_t *screen
)
{
    left_eye = lv_obj_create(
        screen
    );

    style_neon_outline(
        left_eye,
        122,
        62
    );

    lv_obj_align(
        left_eye,
        LV_ALIGN_CENTER,
        -92,
        -85
    );

    create_neon_pupil(
        left_eye
    );

    right_eye = lv_obj_create(
        screen
    );

    style_neon_outline(
        right_eye,
        122,
        62
    );

    lv_obj_align(
        right_eye,
        LV_ALIGN_CENTER,
        92,
        -85
    );

    create_neon_pupil(
        right_eye
    );

    mouth = lv_obj_create(
        screen
    );

    lv_obj_set_size(
        mouth,
        92,
        10
    );

    lv_obj_set_style_bg_color(
        mouth,
        lv_color_hex(FACE_NEON_COLOR),
        LV_PART_MAIN
    );

    lv_obj_set_style_bg_opa(
        mouth,
        LV_OPA_COVER,
        LV_PART_MAIN
    );

    lv_obj_set_style_border_width(
        mouth,
        0,
        LV_PART_MAIN
    );

    lv_obj_set_style_radius(
        mouth,
        LV_RADIUS_CIRCLE,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_color(
        mouth,
        lv_color_hex(FACE_NEON_COLOR),
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_width(
        mouth,
        18,
        LV_PART_MAIN
    );

    lv_obj_set_style_shadow_opa(
        mouth,
        180,
        LV_PART_MAIN
    );

    lv_obj_clear_flag(
        mouth,
        LV_OBJ_FLAG_SCROLLABLE
    );

    lv_obj_align(
        mouth,
        LV_ALIGN_CENTER,
        0,
        25
    );
}


static void display_set_text(const char *text)
{
    if (!bsp_display_lock(1000)) {
        ESP_LOGE(TAG, "Unable to lock LVGL");
        return;
    }

    lv_label_set_text(message_label, text);

    lv_obj_align(
        message_label,
        LV_ALIGN_BOTTOM_MID,
        0,
        -18
    );

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

    create_face(
        screen
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

    lv_obj_align(
        message_label,
        LV_ALIGN_BOTTOM_MID,
        0,
        -18
    );

    bsp_display_unlock();

    ESP_LOGI(TAG, "Display ready");

    run_serial_receiver();
}
