# Third-party notices

Fripouille's original code and documentation are licensed under the MIT
License in `LICENSE`. That license does not replace or override the licenses
of third-party components, tools, or assets described below.

## Content redistributed in this repository

### Montserrat Medium font data

- **Use:** glyph bitmaps and metrics for the ESP32 display font.
- **Location:**
  `firmware/fripouille_esp32/main/fonts/fripouille_font_20.c`.
- **Provenance:** generated from `Montserrat-Medium.ttf` shipped in LVGL
  9.3.0 under `scripts/built_in_font/`. The generated file records that exact
  source path and its selected Unicode ranges.
- **Original project:**
  [The Montserrat Font Project](https://github.com/JulietaUla/Montserrat).
- **Classification:** generated code containing a subset of a third-party
  font asset. The TTF itself is not tracked, but its derived glyph data is
  redistributed.
- **License:** SIL Open Font License, Version 1.1 (`OFL-1.1`).

The LVGL font-generation script uses
[`lv_font_conv`](https://github.com/lvgl/lv_font_conv), an MIT-licensed tool.
The exact converter version used for this generated file was not recorded.
The tool itself is not redistributed in this repository.

The copyright and license text accompanying the exact Montserrat source file
used locally are reproduced below.

> Copyright 2011 The Montserrat Project Authors
> (https://github.com/JulietaUla/Montserrat)

### SIL Open Font License, Version 1.1

```text
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

## Firmware dependencies referenced but not redistributed

The firmware build uses the following external sources. ESP-IDF is installed
separately; ESP-IDF Component Manager downloads the other components. Their
source trees are excluded by `.gitignore`; only `idf_component.yml` and
`dependencies.lock` are tracked here.

| Component | Locked version | Use | Upstream license |
| --- | --- | --- | --- |
| [ESP-IDF](https://github.com/espressif/esp-idf) | 5.5.2 | Firmware framework and toolchain integration | Apache-2.0; ESP-IDF also identifies separately licensed bundled portions |
| [LVGL](https://github.com/lvgl/lvgl/tree/v9.3.0) | 9.3.0 | Embedded display and UI library | MIT |
| [Waveshare ESP32-S3-Touch-LCD-4 BSP](https://components.espressif.com/components/waveshare/esp32_s3_touch_lcd_4/versions/2.0.0) | 2.0.0 | Board initialization and display support | Apache-2.0 |
| [Espressif CMake utilities](https://components.espressif.com/components/espressif/cmake_utilities/versions/0.5.3) | 0.5.3 | Transitive build utilities | Apache-2.0 |
| [Espressif IO expander](https://components.espressif.com/components/espressif/esp_io_expander/versions/1.2.1) | 1.2.1 | Transitive IO-expander abstraction | Apache-2.0 |
| [ESP LCD panel IO additions](https://components.espressif.com/components/espressif/esp_lcd_panel_io_additions/versions/1.0.1~1) | 1.0.1~1 | Transitive display IO | Apache-2.0 |
| [ESP LCD ST7701](https://components.espressif.com/components/espressif/esp_lcd_st7701/versions/2.0.2~2) | 2.0.2~2 | Transitive display controller | Apache-2.0 |
| [ESP LCD Touch](https://components.espressif.com/components/espressif/esp_lcd_touch/versions/1.2.1) | 1.2.1 | Transitive touch abstraction | Apache-2.0 |
| [ESP LCD Touch GT911](https://components.espressif.com/components/espressif/esp_lcd_touch_gt911/versions/1.2.1) | 1.2.1 | Transitive touch-controller driver | Apache-2.0 |
| [ESP LVGL port](https://components.espressif.com/components/espressif/esp_lvgl_port/versions/2.9.0) | 2.9.0 | Transitive LVGL integration | Apache-2.0 |
| [Waveshare CH32V003 IO expander](https://components.espressif.com/components/waveshare/custom_io_expander_ch32v003/versions/2.0.0) | 2.0.0 | Transitive board IO-expander driver | Apache-2.0 |

The license files and SPDX headers in the locally downloaded copies were also
checked against these metadata. Anyone distributing a compiled firmware image
must preserve all notices required by the exact dependency sources included in
that image; this repository-level notice is not a substitute for a binary
distribution audit.

## Other external tools and services

- `setuptools` is referenced as the Python build backend but is not vendored.
- Python, Ollama, and the configured Qwen model are installed separately and
  are not redistributed by this repository. Their own terms apply.
- GitHub Actions workflows reference official actions remotely; their source
  is not copied into this repository.
