# Per-premise re-answering of MathVista loss cases

Model: `Qwen/Qwen3.5-9B` | 10 cases | one prompt **per premise** (9 per case)
Cases fixed by at least one premise: **2/10**

Per-mode correct counts: `1_salient`=0, `2_subtle`=1, `3_spatial`=1, `4_ocr`=0, `5_distractor`=1, `6_alternative`=0, `7_uncertainty`=1, `8_people`=2, `9_digits`=0

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

## pid 19  (GT: `400` | orig wrong: `600.0`)
**Q:** what is the highest amount this class measures?
fixed by: none

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

## pid 20  (GT: `No` | orig wrong: `Yes`)
**Q:** Is the sum of smallest two bar is greater then the largest bar?
fixed by: none

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `Yes` |  |
| 2_subtle | `Yes` |  |
| 3_spatial | `Yes` |  |
| 4_ocr | `Yes` |  |
| 5_distractor | `Yes` |  |
| 6_alternative | `Yes` |  |
| 7_uncertainty | `Yes` |  |
| 8_people | `Yes` |  |
| 9_digits | `Yes` |  |

## pid 27  (GT: `11` | orig wrong: `23.0`)
**Q:** What is the age gap between these two people in image?
fixed by: ['2_subtle', '3_spatial', '5_distractor', '7_uncertainty', '8_people']

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

## pid 37  (GT: `3` | orig wrong: `2.0`)
**Q:** What is the limit as x approaches -1?
fixed by: none

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

## pid 50  (GT: `The population of predatory insects would decrease.` | orig wrong: `The population of predatory insects would increase.`)
**Q:** Based on the food web, what would likely happen if the number of large roach would decrease?
fixed by: ['8_people']

| mode | answer | correct |
|------|--------|---------|
| 1_salient | `The population of predatory insects woul` |  |
| 2_subtle | `The population of predatory insects woul` |  |
| 3_spatial | `The population of predatory insects woul` |  |
| 4_ocr | `The population of predatory insects woul` |  |
| 5_distractor | `The population of predatory insects woul` |  |
| 6_alternative | `The population of predatory insects woul` |  |
| 7_uncertainty | `The population of predatory insects woul` |  |
| 8_people | `The population of predatory insects woul` | ✅ |
| 9_digits | `The population of predatory insects woul` |  |

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
