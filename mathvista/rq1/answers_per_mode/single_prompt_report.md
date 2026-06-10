# RQ1 — single prompt per mode (MathVista loss cases)

Model: `Qwen/Qwen3.5-9B` | 8 cases | one prompt **per mode** (20 per case), one model call each — no separate premise step
Cases fixed by at least one mode: **5/8**

Per-mode correct counts: `1_salient`=0, `2_subtle`=0, `3_spatial`=1, `4_ocr`=1, `5_distractor`=1, `6_alternative`=0, `7_uncertainty`=1, `8_people`=3, `9_digits`=1, `10_point_on_curve`=1, `11_value_curve_reaches`=3, `12_marker_on_vs_off`=1, `13_trace_to_x`=1, `14_read_y_by_ticks`=2, `15_curve_height`=0, `16_two_points_pick_curve`=2, `17_settle_value`=1, `18_coordinate_on_curve`=2, `19_approach_height`=1, `20_hole_on_curve`=1

## pid 7  (GT: `quarter` | orig wrong: `quarter past`)
**Q:** What time is shown? Answer by typing a time word, not a number. It is (_) after eight.
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `quarter past` |  |
| 2_subtle | `quarter past` |  |
| 3_spatial | `quarter past` |  |
| 4_ocr | `quarter past` |  |
| 5_distractor | `quarter past` |  |
| 6_alternative | `quarter past` |  |
| 7_uncertainty | `quarter past` |  |
| 8_people | `quarter past` |  |
| 9_digits | `quarter past` |  |
| 10_point_on_curve | `quarter past` |  |
| 11_value_curve_reaches | `quarter past` |  |
| 12_marker_on_vs_off | `quarter past` |  |
| 13_trace_to_x | `quarter past` |  |
| 14_read_y_by_ticks | `quarter past` |  |
| 15_curve_height | `quarter past` |  |
| 16_two_points_pick_curve | `quarter past` |  |
| 17_settle_value | `quarter past` |  |
| 18_coordinate_on_curve | `quarter past` |  |
| 19_approach_height | `quarter past` |  |
| 20_hole_on_curve | `quarter past` |  |

## pid 19  (GT: `400` | orig wrong: `600.0`)
**Q:** what is the highest amount this class measures?
fixed by: ['8_people', '9_digits', '11_value_curve_reaches', '12_marker_on_vs_off', '13_trace_to_x', '16_two_points_pick_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `600.0` |  |
| 2_subtle | `600.0` |  |
| 3_spatial | `600.0` |  |
| 4_ocr | `600.0` |  |
| 5_distractor | `600.0` |  |
| 6_alternative | `600.0` |  |
| 7_uncertainty | `600.0` |  |
| 8_people | `400.0` | ✅ |
| 9_digits | `400.0` | ✅ |
| 10_point_on_curve | `600.0` |  |
| 11_value_curve_reaches | `400.0` | ✅ |
| 12_marker_on_vs_off | `400.0` | ✅ |
| 13_trace_to_x | `400.0` | ✅ |
| 14_read_y_by_ticks | `600.0` |  |
| 15_curve_height | `600.0` |  |
| 16_two_points_pick_curve | `400.0` | ✅ |
| 17_settle_value | `600.0` |  |
| 18_coordinate_on_curve | `600.0` |  |
| 19_approach_height | `600.0` |  |
| 20_hole_on_curve | `600.0` |  |

## pid 27  (GT: `11` | orig wrong: `23.0`)
**Q:** What is the age gap between these two people in image?
fixed by: ['3_spatial', '4_ocr', '5_distractor', '7_uncertainty', '8_people', '10_point_on_curve', '11_value_curve_reaches', '14_read_y_by_ticks', '17_settle_value', '18_coordinate_on_curve', '19_approach_height', '20_hole_on_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `2.0` |  |
| 2_subtle | `31.0` |  |
| 3_spatial | `11.0` | ✅ |
| 4_ocr | `11.0` | ✅ |
| 5_distractor | `11.0` | ✅ |
| 6_alternative | `2.0` |  |
| 7_uncertainty | `11.0` | ✅ |
| 8_people | `11.0` | ✅ |
| 9_digits | `2.0` |  |
| 10_point_on_curve | `11.0` | ✅ |
| 11_value_curve_reaches | `11.0` | ✅ |
| 12_marker_on_vs_off | `2009.0` |  |
| 13_trace_to_x | `28.0` |  |
| 14_read_y_by_ticks | `11.0` | ✅ |
| 15_curve_height | `31.0` |  |
| 16_two_points_pick_curve | `2009.0` |  |
| 17_settle_value | `11.0` | ✅ |
| 18_coordinate_on_curve | `11.0` | ✅ |
| 19_approach_height | `11.0` | ✅ |
| 20_hole_on_curve | `11.0` | ✅ |

## pid 37  (GT: `3` | orig wrong: `2.0`)
**Q:** What is the limit as x approaches -1?
fixed by: ['14_read_y_by_ticks']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `2.0` |  |
| 2_subtle | `2.0` |  |
| 3_spatial | `2.0` |  |
| 4_ocr | `2.0` |  |
| 5_distractor | `2.0` |  |
| 6_alternative | `2.0` |  |
| 7_uncertainty | `2.0` |  |
| 8_people | `2.0` |  |
| 9_digits | `2.0` |  |
| 10_point_on_curve | `2.0` |  |
| 11_value_curve_reaches | `2.0` |  |
| 12_marker_on_vs_off | `2.0` |  |
| 13_trace_to_x | `2.0` |  |
| 14_read_y_by_ticks | `3.0` | ✅ |
| 15_curve_height | `2.0` |  |
| 16_two_points_pick_curve | `2.0` |  |
| 17_settle_value | `2.0` |  |
| 18_coordinate_on_curve | `2.0` |  |
| 19_approach_height | `2.0` |  |
| 20_hole_on_curve | `2.0` |  |

## pid 41  (GT: `7` | orig wrong: `13.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `20.0` |  |
| 2_subtle | `15.0` |  |
| 3_spatial | `20.0` |  |
| 4_ocr | `26.0` |  |
| 5_distractor | `15.0` |  |
| 6_alternative | `13.0` |  |
| 7_uncertainty | `20.0` |  |
| 8_people | `18.0` |  |
| 9_digits | `21.0` |  |
| 10_point_on_curve | `13.0` |  |
| 11_value_curve_reaches | `11.0` |  |
| 12_marker_on_vs_off | `13.0` |  |
| 13_trace_to_x | `11.0` |  |
| 14_read_y_by_ticks | `11.0` |  |
| 15_curve_height | `12.0` |  |
| 16_two_points_pick_curve | `13.0` |  |
| 17_settle_value | `15.0` |  |
| 18_coordinate_on_curve | `15.0` |  |
| 19_approach_height | `13.0` |  |
| 20_hole_on_curve | `20.0` |  |

## pid 42  (GT: `8` | orig wrong: `0.0`)
**Q:** What is the age gap between these two people in image?
fixed by: ['18_coordinate_on_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `0.0` |  |
| 2_subtle | `7.0` |  |
| 3_spatial | `1.0` |  |
| 4_ocr | `3.0` |  |
| 5_distractor | `0.0` |  |
| 6_alternative | `5.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `1.0` |  |
| 9_digits | `10.0` |  |
| 10_point_on_curve | `3.0` |  |
| 11_value_curve_reaches | `10.0` |  |
| 12_marker_on_vs_off | `3.0` |  |
| 13_trace_to_x | `3.0` |  |
| 14_read_y_by_ticks | `10.0` |  |
| 15_curve_height | `3.0` |  |
| 16_two_points_pick_curve | `5.0` |  |
| 17_settle_value | `5.0` |  |
| 18_coordinate_on_curve | `8.0` | ✅ |
| 19_approach_height | `3.0` |  |
| 20_hole_on_curve | `4.0` |  |

## pid 53  (GT: `1` | orig wrong: `8.0`)
**Q:** What is the age gap between these two people in image?
fixed by: ['8_people', '11_value_curve_reaches', '16_two_points_pick_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `0.0` |  |
| 2_subtle | `3.0` |  |
| 3_spatial | `0.0` |  |
| 4_ocr | `7.0` |  |
| 5_distractor | `6.0` |  |
| 6_alternative | `2.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `1.0` | ✅ |
| 9_digits | `7.0` |  |
| 10_point_on_curve | `7.0` |  |
| 11_value_curve_reaches | `1.0` | ✅ |
| 12_marker_on_vs_off | `0.0` |  |
| 13_trace_to_x | `13.0` |  |
| 14_read_y_by_ticks | `5.0` |  |
| 15_curve_height | `7.0` |  |
| 16_two_points_pick_curve | `1.0` | ✅ |
| 17_settle_value | `13.0` |  |
| 18_coordinate_on_curve | `2.0` |  |
| 19_approach_height | `7.0` |  |
| 20_hole_on_curve | `0.0` |  |

## pid 60  (GT: `22` | orig wrong: `0.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `30.0` |  |
| 2_subtle | `30.0` |  |
| 3_spatial | `25.0` |  |
| 4_ocr | `25.0` |  |
| 5_distractor | `0.0` |  |
| 6_alternative | `15.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `30.0` |  |
| 9_digits | `0.0` |  |
| 10_point_on_curve | `30.0` |  |
| 11_value_curve_reaches | `0.0` |  |
| 12_marker_on_vs_off | `11.0` |  |
| 13_trace_to_x | `0.0` |  |
| 14_read_y_by_ticks | `30.0` |  |
| 15_curve_height | `15.0` |  |
| 16_two_points_pick_curve | `35.0` |  |
| 17_settle_value | `10.0` |  |
| 18_coordinate_on_curve | `0.0` |  |
| 19_approach_height | `30.0` |  |
| 20_hole_on_curve | `12.0` |  |
