# Per-premise re-answering of MathVista loss cases

Model: `Qwen/Qwen3.5-9B` | 8 cases | one prompt **per premise** (20 per case)
Cases fixed by at least one premise: **3/8**

Per-mode correct counts: `1_salient`=0, `2_subtle`=1, `3_spatial`=1, `4_ocr`=0, `5_distractor`=1, `6_alternative`=0, `7_uncertainty`=1, `8_people`=1, `9_digits`=0, `10_point_on_curve`=2, `11_value_curve_reaches`=1, `12_marker_on_vs_off`=0, `13_trace_to_x`=1, `14_read_y_by_ticks`=0, `15_curve_height`=1, `16_two_points_pick_curve`=0, `17_settle_value`=1, `18_coordinate_on_curve`=1, `19_approach_height`=1, `20_hole_on_curve`=3

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
fixed by: ['10_point_on_curve', '13_trace_to_x', '17_settle_value', '19_approach_height', '20_hole_on_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `600.0` |  |
| 2_subtle | `600.0` |  |
| 3_spatial | `600.0` |  |
| 4_ocr | `600.0` |  |
| 5_distractor | `600.0` |  |
| 6_alternative | `600.0` |  |
| 7_uncertainty | `600.0` |  |
| 8_people | `600.0` |  |
| 9_digits | `600.0` |  |
| 10_point_on_curve | `400.0` | ✅ |
| 11_value_curve_reaches | `600.0` |  |
| 12_marker_on_vs_off | `600.0` |  |
| 13_trace_to_x | `400.0` | ✅ |
| 14_read_y_by_ticks | `600.0` |  |
| 15_curve_height | `600.0` |  |
| 16_two_points_pick_curve | `600.0` |  |
| 17_settle_value | `400.0` | ✅ |
| 18_coordinate_on_curve | `600.0` |  |
| 19_approach_height | `400.0` | ✅ |
| 20_hole_on_curve | `400.0` | ✅ |

## pid 27  (GT: `11` | orig wrong: `23.0`)
**Q:** What is the age gap between these two people in image?
fixed by: ['2_subtle', '3_spatial', '5_distractor', '7_uncertainty', '8_people', '10_point_on_curve', '11_value_curve_reaches', '15_curve_height', '18_coordinate_on_curve', '20_hole_on_curve']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `27.0` |  |
| 2_subtle | `11.0` | ✅ |
| 3_spatial | `11.0` | ✅ |
| 4_ocr | `31.0` |  |
| 5_distractor | `11.0` | ✅ |
| 6_alternative | `18.0` |  |
| 7_uncertainty | `11.0` | ✅ |
| 8_people | `11.0` | ✅ |
| 9_digits | `31.0` |  |
| 10_point_on_curve | `11.0` | ✅ |
| 11_value_curve_reaches | `11.0` | ✅ |
| 12_marker_on_vs_off | `31.0` |  |
| 13_trace_to_x | `27.0` |  |
| 14_read_y_by_ticks | `27.0` |  |
| 15_curve_height | `11.0` | ✅ |
| 16_two_points_pick_curve | `31.0` |  |
| 17_settle_value | `31.0` |  |
| 18_coordinate_on_curve | `11.0` | ✅ |
| 19_approach_height | `31.0` |  |
| 20_hole_on_curve | `11.0` | ✅ |

## pid 37  (GT: `3` | orig wrong: `2.0`)
**Q:** What is the limit as x approaches -1?
fixed by: ['20_hole_on_curve']

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
| 14_read_y_by_ticks | `2.0` |  |
| 15_curve_height | `2.0` |  |
| 16_two_points_pick_curve | `2.0` |  |
| 17_settle_value | `2.0` |  |
| 18_coordinate_on_curve | `2.0` |  |
| 19_approach_height | `2.0` |  |
| 20_hole_on_curve | `3.0` | ✅ |

## pid 41  (GT: `7` | orig wrong: `13.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `11.0` |  |
| 2_subtle | `11.0` |  |
| 3_spatial | `11.0` |  |
| 4_ocr | `1.0` |  |
| 5_distractor | `11.0` |  |
| 6_alternative | `11.0` |  |
| 7_uncertainty | `11.0` |  |
| 8_people | `10.0` |  |
| 9_digits | `None` |  |
| 10_point_on_curve | `11.0` |  |
| 11_value_curve_reaches | `11.0` |  |
| 12_marker_on_vs_off | `11.0` |  |
| 13_trace_to_x | `11.0` |  |
| 14_read_y_by_ticks | `0.0` |  |
| 15_curve_height | `11.0` |  |
| 16_two_points_pick_curve | `11.0` |  |
| 17_settle_value | `11.0` |  |
| 18_coordinate_on_curve | `11.0` |  |
| 19_approach_height | `11.0` |  |
| 20_hole_on_curve | `11.0` |  |

## pid 42  (GT: `8` | orig wrong: `0.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `5.0` |  |
| 2_subtle | `0.0` |  |
| 3_spatial | `5.0` |  |
| 4_ocr | `0.0` |  |
| 5_distractor | `0.0` |  |
| 6_alternative | `5.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `0.0` |  |
| 9_digits | `None` |  |
| 10_point_on_curve | `5.0` |  |
| 11_value_curve_reaches | `5.0` |  |
| 12_marker_on_vs_off | `5.0` |  |
| 13_trace_to_x | `5.0` |  |
| 14_read_y_by_ticks | `0.0` |  |
| 15_curve_height | `1.0` |  |
| 16_two_points_pick_curve | `5.0` |  |
| 17_settle_value | `5.0` |  |
| 18_coordinate_on_curve | `5.0` |  |
| 19_approach_height | `5.0` |  |
| 20_hole_on_curve | `0.0` |  |

## pid 53  (GT: `1` | orig wrong: `8.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `5.0` |  |
| 2_subtle | `5.0` |  |
| 3_spatial | `3.0` |  |
| 4_ocr | `0.0` |  |
| 5_distractor | `0.0` |  |
| 6_alternative | `2.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `0.0` |  |
| 9_digits | `None` |  |
| 10_point_on_curve | `0.0` |  |
| 11_value_curve_reaches | `0.0` |  |
| 12_marker_on_vs_off | `2.0` |  |
| 13_trace_to_x | `2.0` |  |
| 14_read_y_by_ticks | `0.0` |  |
| 15_curve_height | `0.0` |  |
| 16_two_points_pick_curve | `2.0` |  |
| 17_settle_value | `2.0` |  |
| 18_coordinate_on_curve | `0.0` |  |
| 19_approach_height | `0.0` |  |
| 20_hole_on_curve | `0.0` |  |

## pid 60  (GT: `22` | orig wrong: `0.0`)
**Q:** What is the age gap between these two people in image?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `30.0` |  |
| 2_subtle | `20.0` |  |
| 3_spatial | `25.0` |  |
| 4_ocr | `30.0` |  |
| 5_distractor | `25.0` |  |
| 6_alternative | `25.0` |  |
| 7_uncertainty | `0.0` |  |
| 8_people | `15.0` |  |
| 9_digits | `0.0` |  |
| 10_point_on_curve | `20.0` |  |
| 11_value_curve_reaches | `30.0` |  |
| 12_marker_on_vs_off | `10.0` |  |
| 13_trace_to_x | `20.0` |  |
| 14_read_y_by_ticks | `25.0` |  |
| 15_curve_height | `20.0` |  |
| 16_two_points_pick_curve | `10.0` |  |
| 17_settle_value | `20.0` |  |
| 18_coordinate_on_curve | `20.0` |  |
| 19_approach_height | `20.0` |  |
| 20_hole_on_curve | `25.0` |  |
